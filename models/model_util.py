""" Module for creating models. """

from math import prod
from typing import Tuple

import torch
from cuda_selector import auto_cuda  # type: ignore

from models.classifier_base import ClassifierBase
from models.mlp import MLP, MLPFeature


def get_model_arch(algo_name: str) -> str:
    """ Get model arch. """

    if "valen" in algo_name or "pico" in algo_name or "cel" in algo_name:
        return "mlpfeat"

    return "mlp"


def create_model(
    arch: str, num_class: int, input_shape: Tuple,
) -> ClassifierBase:
    """ Create a model with the given architecture. """

    torch.set_float32_matmul_precision("high")
    if arch == "mlp":
        # MLP with flattened input
        m_features = prod(input_shape)
        model: ClassifierBase = MLP(m_features, num_class)
    elif arch == "mlpfeat":
        # MLP with latent features
        m_features = prod(input_shape)
        model = MLPFeature(m_features, num_class)
    else:
        raise ValueError(f"Invalid architecture '{arch}'.")

    return model


def get_device() -> torch.device:
    """ Get a CUDA device """

    if torch.cuda.is_available():
        device = torch.device(auto_cuda("utilization", fallback=False))
    else:
        device = torch.device("cpu")
    return device
