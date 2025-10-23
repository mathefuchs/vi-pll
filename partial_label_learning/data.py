""" Module for loading data. """

import io
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
import torchvision  # type: ignore
from cuda_selector import auto_cuda  # type: ignore
from scipy.io import loadmat  # type: ignore
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from partial_label_learning.config import REAL_WORLD_LABEL_TO_PATH


def random_class_imbalance(num_classes: int) -> np.ndarray:
    """ Get a random vector of importances per class """

    # Determine sample size of each class
    exp_decay = 0.025 ** (1 / num_classes)
    sample_frac = np.array([
        exp_decay ** (i + 1)
        for i in range(num_classes)
    ])
    np.random.shuffle(sample_frac)
    return sample_frac


def _augment_targets_instance_dependent(
    x_data: np.ndarray, y_data: np.ndarray,
    hardness: Tuple[float, float], class_imb: np.ndarray,
) -> np.ndarray:
    """ Augments a supervised dataset with instance-dependent noise. """

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(auto_cuda("utilization"))
    else:
        device = torch.device("cpu")

    # Preprocess data
    x_full = x_data.reshape(x_data.shape[0], -1)
    x_min = np.min(x_full, axis=0)
    x_max = np.max(x_full, axis=0)
    diff = np.clip(x_max - x_min, a_min=1e-10, a_max=None)
    x_full = (x_full - x_min) / diff

    # Create model
    model = nn.Sequential(
        nn.Linear(x_full.shape[1], 300), nn.ReLU(),
        nn.Linear(300, y_data.shape[1]), nn.Softmax(dim=1),
    )
    model.to(device)
    optim = torch.optim.SGD(model.parameters())
    model.train()

    # Prepare data
    x_train_tensor = torch.tensor(x_full, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_data, dtype=torch.float32)
    data_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=64, shuffle=True, drop_last=True,
    )

    # Training loop
    for _ in range(20):
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            probs = model(x_batch)
            loss = torch.mean(torch.sum(
                (y_batch - probs) ** 2, dim=1))
            optim.zero_grad()
            loss.backward()
            optim.step()

    # Inference
    model.eval()
    inference_loader = DataLoader(
        TensorDataset(torch.tensor(x_full, dtype=torch.float32)),
        batch_size=64, shuffle=False,
    )
    with torch.no_grad():
        all_results = []
        for x_batch in inference_loader:
            x_batch = x_batch[0].to(device)
            all_results.append(model(x_batch).cpu().numpy())
        train_probs = np.vstack(all_results)

    # Determine augmentation probabilities
    train_false_probs = (1 - y_data) * train_probs
    max_false_probs = np.max(train_false_probs, axis=1, keepdims=True)
    train_false_probs /= np.where(
        max_false_probs > 1e-10, max_false_probs, 1.0)
    mean_false_probs = np.mean(
        train_false_probs, axis=1, keepdims=True)
    train_false_probs = train_false_probs / np.where(
        mean_false_probs > 1e-10, mean_false_probs, 1.0)
    class_imb = class_imb / np.max(class_imb)
    train_false_probs = (
        0.3 * hardness[0] * train_false_probs +
        0.7 * hardness[1] * class_imb
    )
    train_false_probs = np.clip(train_false_probs, 0.001, 0.999)

    # Augmentation
    sampler = torch.distributions.binomial.Binomial(
        total_count=1, probs=torch.tensor(train_false_probs))
    sample = sampler.sample()
    y_full_copy = y_data.copy()
    y_full_copy[sample == 1] = 1

    return y_full_copy


def _make_imbalanced(
    y_true: np.ndarray, num_classes: int,
    class_imb: np.ndarray,
) -> np.ndarray:
    """ Returns indices describing which instances to use. """

    # Subsample
    final_indices = []
    for cl in range(num_classes):
        cl_indices = np.arange(y_true.shape[0])[y_true == cl]
        cl_choices = np.random.choice(
            cl_indices, size=int(0.3 * class_imb[cl] * cl_indices.shape[0]),
            replace=False,
        )
        final_indices.append(cl_choices)

    # Return mask
    final_ind = np.concat(final_indices, dtype=int)
    np.random.shuffle(final_ind)
    return final_ind


class Dataset:
    """ A dataset. """

    def __init__(
        self, x_full: np.ndarray, y_full: np.ndarray, y_true: np.ndarray,
    ) -> None:
        self.x_full = x_full
        self.y_full = y_full
        self.y_true = y_true

    def augment_targets_instance_dependent(
        self, hardness: Tuple[float, float], class_imb: np.ndarray,
    ) -> "Dataset":
        """ Add instance-dependent noise. """

        return Dataset(
            self.x_full,
            _augment_targets_instance_dependent(
                self.x_full, self.y_full, hardness, class_imb),
            self.y_true,
        )

    def make_imbalanced(
        self, class_imb: np.ndarray,
    ) -> "Dataset":
        """ Make dataset imbalanced. """

        indices = _make_imbalanced(
            self.y_true, self.y_full.shape[1], class_imb)
        return Dataset(
            self.x_full[indices].copy(),
            self.y_full[indices].copy(),
            self.y_true[indices].copy(),
        )


class Datasplit:
    """ A data split. """

    def __init__(
        self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray,
        y_true_train: np.ndarray, y_true_test: np.ndarray,
    ) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_true_train = y_true_train
        self.y_true_test = y_true_test

    def augment_targets_instance_dependent(
        self, hardness: Tuple[float, float], class_imb: np.ndarray,
    ) -> "Datasplit":
        """ Add instance-dependent noise. """

        return Datasplit(
            self.x_train, self.x_test,
            _augment_targets_instance_dependent(
                self.x_train, self.y_train, hardness, class_imb),
            self.y_true_train, self.y_true_test,
        )

    def make_imbalanced(
        self, class_imb: np.ndarray,
    ) -> "Datasplit":
        """ Make dataset imbalanced. """

        indices = _make_imbalanced(
            self.y_true_train, self.y_train.shape[1], class_imb)
        return Datasplit(
            self.x_train[indices].copy(),
            self.x_test,
            self.y_train[indices].copy(),
            self.y_true_train[indices].copy(),
            self.y_true_test,
        )

    def store_to_file(self, path: str) -> None:
        """ Store datasplit to path. """

        np.savez_compressed(
            path, x_train=self.x_train, x_test=self.x_test,
            y_train=self.y_train, y_true_train=self.y_true_train,
            y_true_test=self.y_true_test,
        )

    @classmethod
    def restore_from_file(cls, path: str) -> "Datasplit":
        """ Restore from file. """

        npz_file = np.load(path)
        return cls(
            npz_file["x_train"], npz_file["x_test"], npz_file["y_train"],
            npz_file["y_true_train"], npz_file["y_true_test"],
        )


def flatten_if_image(inputs: np.ndarray) -> np.ndarray:
    """ Flattens the data if it represents an image. """

    return inputs.reshape(inputs.shape[0], -1).copy()


def get_rl_dataset(dataset_name: str) -> Dataset:
    """ Retrieves a real-world dataset. """

    # Coerce data into dense array
    def coerce(data) -> np.ndarray:
        try:
            return data.toarray()
        except:  # pylint: disable=bare-except
            return data

    # Extract raw data
    raw_mat_data = loadmat(REAL_WORLD_LABEL_TO_PATH[dataset_name])
    x_raw = coerce(raw_mat_data["data"])
    y_partial_raw = coerce(raw_mat_data["partial_target"].transpose())
    y_true_raw = np.argmax(
        coerce(raw_mat_data["target"].transpose()), axis=1)

    available_classes = set(map(int, y_true_raw))
    classes_in_use = []
    for cl in range(y_partial_raw.shape[1]):
        if np.count_nonzero(y_partial_raw[:, cl]) != 0 \
                and cl in available_classes:
            classes_in_use.append(cl)

    # Collect all relevant data
    x_list = []
    y_partial_list = []
    y_true_list: List[int] = []
    mask = np.array(list(sorted(list(classes_in_use))))
    for x_row, y_partial_row, y_true_row in zip(
        x_raw, y_partial_raw, y_true_raw,
    ):
        if int(y_true_row) in classes_in_use:
            x_list.append(x_row)
            y_partial_list.append(y_partial_row[mask])
            y_true_list.append(int(np.where(
                mask == int(y_true_row))[0][0]))
    x_arr = np.array(x_list)
    y_partial_arr = np.array(y_partial_list)
    y_true_arr = np.array(y_true_list)
    x_arr = x_arr[:, x_arr.var(axis=0) > 1e-30].copy()

    # Store dataset
    return Dataset(x_arr, y_partial_arr, y_true_arr)


def get_mnist_dataset(dataset_name: str) -> Datasplit:
    """ Retrieves an MNIST dataset. """

    # Extract datasets
    if dataset_name == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root="./data/image-data", train=True,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data/image-data", train=False,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
    elif dataset_name == "fmnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root="./data/image-data", train=True,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="./data/image-data", train=False,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
    elif dataset_name == "kmnist":
        train_dataset = torchvision.datasets.KMNIST(
            root="./data/image-data", train=True,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
        test_dataset = torchvision.datasets.KMNIST(
            root="./data/image-data", train=False,
            transform=torchvision.transforms.ToTensor(), download=True,
        )
    else:
        raise ValueError()

    # Extract numpy data
    x_train = (train_dataset.data.view(-1, 784).numpy() / 255).copy()
    x_test = (test_dataset.data.view(-1, 784).numpy() / 255).copy()
    y_train_true = train_dataset.targets.numpy().copy()
    y_test_true = test_dataset.targets.numpy().copy()
    l_classes = np.unique(y_train_true).shape[0]
    y_train = np.zeros((x_train.shape[0], l_classes), dtype=int)
    for i, y_val in enumerate(y_train_true):
        y_train[i, y_val] = 1

    # Create dataset
    return Datasplit(
        x_train, x_test, y_train, y_train_true, y_test_true,
    )


def get_vision_dataset(dataset_name: str) -> Datasplit:
    """ Retrieves a vision dataset from 'blip2_feat'. """

    if dataset_name not in ["cifar10", "cifar100"]:
        raise RuntimeError("Dataset unknown.")

    # Read all parts and join them
    parts = []
    for part in sorted(glob(f"./blip2_feat/{dataset_name}_blip2_feat.npz.part[0-9]")):
        with open(part, "rb") as file:
            parts.append(file.read())
    npz_file = np.load(io.BytesIO(b"".join(parts)))

    # Build data
    x_train = npz_file["x_train"]
    y_train_true = npz_file["y_train"]
    x_test = npz_file["x_test"]
    y_test_true = npz_file["y_test"]
    l_classes = np.unique(y_train_true).shape[0]
    y_train = np.zeros((x_train.shape[0], l_classes), dtype=np.int64)
    for i, y_val in enumerate(y_train_true):
        y_train[i, y_val] = 1
    return Datasplit(
        x_train, x_test, y_train, y_train_true, y_test_true,
    )
