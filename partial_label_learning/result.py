""" Module for bundling algorithm results. """

from typing import List

import numpy as np


class SplitResult:
    """ Results on either the train or test set. """

    def __init__(
        self,
        pred: np.ndarray,
        conf_logits: np.ndarray,
        conf_probs: np.ndarray,
        beliefs: np.ndarray,
        uncertainty: np.ndarray,
        reject: np.ndarray,
        is_guessing: np.ndarray,
    ) -> None:
        self.pred = pred
        self.conf_logits = conf_logits
        self.conf_probs = conf_probs
        self.beliefs = beliefs
        self.uncertainty = uncertainty
        self.reject = reject
        self.is_guessing = is_guessing

    @classmethod
    def from_scores(
        cls,
        rng: np.random.Generator,
        conf_scores: np.ndarray,
    ):
        """ Create results from non-negative scores with random tie-breaking. """

        # Extract prediction
        prob_sum = np.sum(conf_scores, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        conf_probs = conf_scores / prob_sum
        pred_list: List[int] = []
        guessing: List[bool] = []
        for score_row in conf_probs:
            max_idx = np.flatnonzero(np.isclose(score_row, np.max(score_row)))
            if max_idx.shape[0] == 1:
                pred_list.append(max_idx[0])
                guessing.append(False)
            elif max_idx.shape[0] > 1:
                pred_list.append(int(rng.choice(max_idx)))
                guessing.append(True)
            else:
                # print("> Warning: Empty selection")
                pred_list.append(-1)
                guessing.append(True)

        # Compute logits for reference
        conf_logits = np.log(
            conf_probs,
            out=-np.inf * np.ones_like(conf_probs),
            where=conf_probs > 1e-10,
        )

        return cls(
            np.array(pred_list), conf_logits, conf_probs, conf_probs,
            np.zeros(len(pred_list), dtype=float),
            np.zeros(len(pred_list), dtype=bool), np.array(guessing),
        )
