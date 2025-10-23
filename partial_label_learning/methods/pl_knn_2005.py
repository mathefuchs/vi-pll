""" Module for PL-KNN. """

from typing import Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors  # type: ignore

from models.classifier_base import ClassifierBase
from partial_label_learning.data import flatten_if_image
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class PlKnn(PllBaseClassifier):
    """
    PL-KNN by Hüllermeier and Beringer,
    "Learning from Ambiguously Labeled Examples."
    """

    def __init__(
        self, rng: np.random.Generator, debug: bool,
        model: ClassifierBase, device: torch.device, **kwargs,
    ) -> None:
        super().__init__(rng, debug, model, device, **kwargs)
        self.n_neighbors = 10
        self.knn: Optional[NearestNeighbors] = None
        self.y_train: Optional[np.ndarray] = None

    def _get_knn_y_pred(
        self, candidates: Optional[np.ndarray],
        nn_dists: np.ndarray, nn_indices: np.ndarray,
    ) -> SplitResult:
        assert self.y_train is not None
        y_voting = np.zeros(
            (nn_indices.shape[0], self.y_train.shape[1]))
        for i, (nn_dist, nn_idx) in enumerate(zip(nn_dists, nn_indices)):
            dist_sum = nn_dist.sum()
            if dist_sum < 1e-6:
                sims = np.ones_like(nn_dist)
            else:
                sims = 1 - nn_dist / dist_sum

            for sim, idx in zip(sims, nn_idx):
                y_voting[i, :] += self.y_train[idx, :] * sim

        if candidates is not None:
            for i in range(y_voting.shape[0]):
                y_voting[i, candidates[i] == 0] = 0

        # Return predictions
        return SplitResult.from_scores(self.rng, y_voting)

    def fit(
        self, inputs: np.ndarray, partial_targets: np.ndarray,
    ) -> SplitResult:
        """ Fits the model to the given inputs.

        Args:
            inputs (np.ndarray): The inputs.
            partial_targets (np.ndarray): The partial targets.

        Returns:
            SplitResult: The disambiguated targets.
        """

        inputs = flatten_if_image(inputs)
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1)
        self.knn.fit(inputs)
        self.y_train = partial_targets
        return self._get_knn_y_pred(partial_targets, *self.knn.kneighbors())

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        assert self.knn is not None
        inputs = flatten_if_image(inputs)
        return self._get_knn_y_pred(
            None, *self.knn.kneighbors(inputs))
