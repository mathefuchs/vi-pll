""" Module for VI. """

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class ViAblation(PllBaseClassifier):
    """ Variational Inference """

    def fit(
        self, inputs: np.ndarray, partial_targets: np.ndarray,
    ) -> SplitResult:
        """ Fits the model to the given inputs. """

        # Data
        x_train_t = torch.tensor(inputs, dtype=torch.float32)
        y_train_t = torch.tensor(partial_targets, dtype=torch.float32)
        train_indices = torch.arange(inputs.shape[0], dtype=torch.int32)
        p_y_x = torch.tensor(partial_targets, dtype=torch.float32)
        p_y_x /= p_y_x.sum(dim=1, keepdim=True)
        data_loader = DataLoader(
            TensorDataset(train_indices, x_train_t, y_train_t, p_y_x),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters())

        # Training loop
        self.model.train()
        for epoch in range(1001):
            losses = []
            for idx, x_batch, s_batch, p_y_x_batch in data_loader:
                x_batch = x_batch.to(self.device)
                s_batch = s_batch.to(self.device)
                p_y_x_batch = p_y_x_batch.to(self.device)

                # Get Q(Y | X)
                # pylint: disable=not-callable
                alpha = 1 + nn.functional.softplus(self.model(x_batch)[1])
                dirichlet_dist = torch.distributions.Dirichlet(alpha)
                q_y_x_sample = dirichlet_dist.rsample((10,))

                # E_{Y ~ Q(Y | X)}[ log Q(Y | X) ]
                log_q_y_x = torch.mean(torch.sum(
                    q_y_x_sample * torch.log(q_y_x_sample + 1e-10), dim=-1,
                ), dim=0)

                # E_{Y ~ Q(Y | X)}[ log P(Y | X) ]
                log_p_y_x = torch.mean(torch.sum(
                    q_y_x_sample * torch.log(p_y_x_batch + 1e-10), dim=-1,
                ), dim=0)

                # E_{Y ~ Q(Y | X)}[ log P(S | X, Y) ];
                # (x_batch, s_batch) -> Matrix (batch, classes): p_s_xy per class;
                # P(S | Y) = 2^{1 - k}, if Y \in S, else 0
                p_s_xy_per_class = s_batch
                log_p_s_xy = torch.mean(torch.sum(
                    q_y_x_sample * torch.log(p_s_xy_per_class + 1e-10), dim=-1,
                ), dim=0)

                # Loss
                loss = torch.mean(log_q_y_x - log_p_y_x - log_p_s_xy)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update P(Y | X)
                with torch.no_grad():
                    q_y_x_batch = alpha / torch.sum(
                        alpha, dim=1, keepdim=True)
                    new_p_y_x = (q_y_x_batch + 1e-10) * s_batch
                    new_p_y_x /= torch.sum(new_p_y_x, dim=1, keepdim=True)
                    p_y_x[idx] = new_p_y_x.cpu()

            if self.debug and epoch % 100 == 0:
                print(
                    f"    Epoch {epoch: >4}: "
                    f"loss {np.mean(losses):.4f}"
                )

        # Return results
        return self.predict(inputs)

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        if self.model is None:
            raise ValueError()

        inference_loader = DataLoader(
            TensorDataset(torch.tensor(inputs, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False,
        )

        # Switch to eval mode
        self.model.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                # pylint: disable=not-callable
                out = 1 + nn.functional.softplus(self.model(x_batch)[1])
                out /= torch.sum(out, dim=1, keepdim=True)
                all_results.append(out.cpu().numpy())
            train_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, train_probs)
