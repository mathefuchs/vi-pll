""" Module for VI. """

import math
import warnings
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from scipy.optimize import minimize  # type: ignore
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.classifier_base import ClassifierBase
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class ConditionalEncoder(nn.Module):
    """ Encoder """

    def __init__(self, input_dim, y_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim + y_dim, hidden_dim), nn.ReLU(),
        )
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of q(z|x, y)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance

    def forward(self, x, y):
        """ Forward pass """

        h = self.fc1(torch.cat((x, y), dim=1))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar


class ConditionalDecoder(nn.Module):
    """ Decoder """

    def __init__(self, latent_dim, y_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.recon = nn.Sequential(
            nn.Linear(latent_dim + y_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Hardsigmoid(),
        )

    def forward(self, z, y):
        """ Forward """

        return self.recon(torch.cat((z, y), dim=-1))


class CVAE(nn.Module):
    """ Conditional Variational Auto-Encoder """

    def __init__(
        self, input_dim, y_dim, latent_dim, hidden_dim,
    ):
        super().__init__()
        self.encoder = ConditionalEncoder(
            input_dim, y_dim, latent_dim, hidden_dim)
        self.decoder = ConditionalDecoder(
            latent_dim, y_dim, input_dim, hidden_dim)

    def forward(self, x, y):
        """ Forward """

        mu, logvar = self.encoder(x, y)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = self.decoder(z, y)
        return x_recon, mu, logvar


class ViPll(PllBaseClassifier):
    """ Variational Inference """

    def __init__(
        self, rng: np.random.Generator, debug: bool,
        model: ClassifierBase, device: torch.device, **kwargs,
    ) -> None:
        super().__init__(rng, debug, model, device, **kwargs)
        self.model2 = deepcopy(self.model)
        self.model2.to(device)
        self.logits_to_evidence = nn.functional.softplus
        self.lmbd = float(kwargs["lmbd"]) if "lmbd" in kwargs else 10.0
        self.prior_coef = float(kwargs["prior"]) if "prior" in kwargs else 0.0
        self.num_classes: int = -1
        self.p_x_y_model: Optional[CVAE] = None
        self.sigma = 0.0

    def _fit_p_x_y_model(
        self, p_x_y_model: CVAE, inputs: np.ndarray,
        partial_targets: np.ndarray,
    ) -> float:
        """ Fit P(X | Y) model. """

        optimizer = torch.optim.AdamW(p_x_y_model.parameters())
        x_train_t = torch.tensor(inputs, dtype=torch.float32)
        s_train_t = torch.tensor(partial_targets, dtype=torch.float32)
        data_loader = DataLoader(
            TensorDataset(x_train_t, s_train_t),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        for epoch in range(501):
            losses = []
            for x_batch, s_batch in data_loader:
                x_batch = x_batch.to(self.device)
                s_batch = s_batch.to(self.device)

                x_recon, mu, logvar = p_x_y_model(x_batch, s_batch)
                recon_loss = nn.functional.mse_loss(x_recon, x_batch)
                kl_div = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
                loss = recon_loss + kl_div

                losses.append(math.sqrt(recon_loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.debug and epoch % 100 == 0:
                print(
                    f"    Epoch {epoch: >4}: "
                    f"P(X | Y) RMSE {np.mean(losses[-10:]):.4f}"
                )

        # Return RMSE as an estimate of the standard deviation
        ema_rmse = losses[0]
        for l in losses[1:]:
            ema_rmse = 0.95 * ema_rmse + 0.05 * l
        return float(ema_rmse)

    def _compute_log_p_x_given_y(
        self, model: CVAE, x: torch.Tensor,
        y: torch.Tensor, num_samples: int, sigma: float,
    ):
        model.eval()
        with torch.no_grad():
            # Sample K times
            mu, logvar = model.encoder(x, y)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(
                (num_samples, *std.shape),
                dtype=torch.float32, device=x.device,
            )
            z = mu + eps * std  # (num_samples, batch, latent_dim)

            # Decode each sample
            x_recons = model.decoder(
                z, y.view(1, *y.shape).repeat(num_samples, 1, 1))

            # IWAE
            mse = ((x.view(1, *x.shape).repeat(
                num_samples, 1, 1) - x_recons) ** 2).mean(dim=-1)
            log_p_x_yz = -0.5 * (
                mse / sigma ** 2 + math.log(2 * math.pi * sigma ** 2))
            log_p_z_y = -0.5 * (z ** 2 + math.log(2 * math.pi)).mean(dim=-1)
            log_q_z_xy = -0.5 * (
                (z - mu) ** 2 / torch.exp(logvar)
                + logvar + math.log(2 * math.pi)
            ).mean(dim=-1)
            weights = log_p_x_yz + 0.01 * (log_p_z_y - log_q_z_xy)
            log_p_x_y = torch.logsumexp(weights, dim=0) - math.log(num_samples)

        return log_p_x_y

    def _max_entropy_p_y(self, partial_targets: np.ndarray) -> np.ndarray:
        """ Find max entropy prior that fulfills constraints. """

        def entropy(x):
            x_clip = np.clip(x, 1e-10, 1)
            return -np.sum(x * np.log(x_clip))

        def constraint_sum(x):
            return np.sum(x) - 1

        num_inst = partial_targets.shape[0]
        min_cnt = partial_targets[partial_targets.sum(1) == 1].sum(0)
        max_cnt = partial_targets.sum(0)
        min_freq = min_cnt / num_inst
        max_freq = max_cnt / num_inst

        # Constraints and bounds
        constraints = [{"type": "eq", "fun": constraint_sum}]
        bounds = list(zip(min_freq, max_freq))
        start_val = 0.5 * (max_freq + min_freq)
        start_val /= np.sum(start_val)

        # Optimization: Maximize entropy (minimize negative entropy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                lambda x: -entropy(x), start_val,
                bounds=bounds, constraints=constraints,
            )
        if not result.success:
            raise RuntimeError("max entropy")

        return result.x

    def fit(
        self, inputs: np.ndarray, partial_targets: np.ndarray,
    ) -> SplitResult:
        """ Fits the model to the given inputs. """

        # P(Y) prior
        num_classes = partial_targets.shape[1]
        self.num_classes = num_classes
        prior = torch.tensor(
            self._max_entropy_p_y(partial_targets),
            dtype=torch.float32, device=self.device,
        )

        # Model for P(X | Y)
        large_scale = inputs.shape[1] == 784
        p_x_y_model = CVAE(
            inputs.shape[1], num_classes,
            latent_dim=128 if large_scale else 32,
            hidden_dim=512 if large_scale else 128,
        )
        p_x_y_model.to(self.device)
        p_x_y_model.compile()
        self.p_x_y_model = p_x_y_model

        # Optimizer
        optimizer = torch.optim.AdamW(
            list(self.model.parameters())
            + list(self.model2.parameters())
        )
        optimizer_pxy = torch.optim.AdamW(
            p_x_y_model.parameters(), lr=1e-4,
        )

        # Fit P(X | Y) model
        sigma = self._fit_p_x_y_model(p_x_y_model, inputs, partial_targets)
        p_x_y_model.eval()

        # Data
        x_train_t = torch.tensor(inputs, dtype=torch.float32)
        s_train_t = torch.tensor(partial_targets, dtype=torch.float32)
        idx = torch.arange(inputs.shape[0], dtype=torch.long)
        curr_y = torch.tensor(partial_targets, dtype=torch.float32)
        data_loader = DataLoader(
            TensorDataset(idx, x_train_t, s_train_t, curr_y),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        # Training loop
        self.model.train()
        self.model2.train()
        for epoch in range(self.num_epoch):
            losses = []
            for idx_batch, x_batch, s_batch, curr_y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                s_batch = s_batch.to(self.device)
                curr_y_batch = curr_y_batch.to(self.device)

                # Get Q(Y | X)
                logits = 0.5 * (
                    self.model(x_batch + 0.01 * torch.randn_like(x_batch))[1] +
                    self.model2(x_batch + 0.01 * torch.randn_like(x_batch))[1]
                )
                # pylint: disable=not-callable
                alpha = 1 + self.logits_to_evidence(logits)
                dirichlet_dist = torch.distributions.Dirichlet(alpha)
                q_y_x_sample = dirichlet_dist.rsample((10,))

                # E_{Y ~ Q(Y | X)}[ log Q(Y | X) ]
                log_q_y_x = torch.mean(torch.sum(
                    q_y_x_sample * torch.log(q_y_x_sample + 1e-10), dim=-1,
                ), dim=0)

                # E_{Y ~ Q(Y | X)}[ log P(Y) ]
                log_p_y = torch.mean(torch.sum(
                    q_y_x_sample * torch.log(prior + 1e-10), dim=-1,
                ), dim=0)

                # E_{Y ~ Q(Y | X)}[ log P(X | Y) ]
                with torch.no_grad():
                    x_rep = torch.repeat_interleave(
                        x_batch, repeats=num_classes, dim=0)
                    y_rep = torch.eye(
                        num_classes, dtype=x_rep.dtype, device=x_rep.device
                    ).repeat((x_batch.shape[0], 1))
                    log_per_x_y = self._compute_log_p_x_given_y(
                        p_x_y_model, x_rep, y_rep, num_samples=10, sigma=sigma)
                    log_likelihood = log_per_x_y.view(
                        x_batch.shape[0], num_classes)
                log_p_x_y = torch.mean(torch.sum(
                    q_y_x_sample * log_likelihood, dim=-1,
                ), dim=0)

                # E_{Y ~ Q(Y | X)}[ log P(S | X, Y) ];
                # P(S | Y) = 1 / 2^{k - 1}, if Y \in S, else 0
                p_s_xy_per_class = curr_y_batch
                log_p_s_xy = torch.mean(torch.sum(
                    q_y_x_sample * (
                        torch.log(p_s_xy_per_class + 1e-10)
                        # - (num_classes - 1) * math.log(2)  # Normalization
                    ), dim=-1,
                ), dim=0)

                # Loss
                loss = torch.mean(
                    log_q_y_x - self.prior_coef * log_p_y
                    - self.lmbd * log_p_x_y - log_p_s_xy
                )

                # Update P(X | Y) model
                p_x_y_model.train()
                x_recon, mu, logvar = p_x_y_model(x_batch, curr_y_batch)
                recon_loss = nn.functional.mse_loss(x_recon, x_batch)
                kl_div = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
                p_x_y_loss = recon_loss + kl_div
                # sigma = 0.95 * sigma + 0.05 * math.sqrt(recon_loss.item())
                loss += p_x_y_loss

                # Log losses
                losses.append({
                    "loss": loss.item(),
                    "pxy_model": math.sqrt(p_x_y_loss.item()),
                    "log_q_y_x": log_q_y_x.mean().item(),
                    "log_p_y": log_p_y.mean().item(),
                    "log_p_x_y": log_p_x_y.mean().item(),
                    "log_p_s_xy": log_p_s_xy.mean().item(),
                })

                # Back-propagation
                optimizer.zero_grad()
                optimizer_pxy.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_pxy.step()

                # Update weights
                with torch.no_grad():
                    q_y_x_batch = alpha / torch.sum(alpha, dim=1, keepdim=True)
                    new_p_y_x = (q_y_x_batch + 1e-10) * s_batch
                    new_p_y_x /= torch.sum(new_p_y_x, dim=1, keepdim=True)
                    s_weighted = new_p_y_x / torch.max(
                        new_p_y_x, dim=1, keepdim=True).values
                    curr_y[idx_batch] = s_weighted.cpu()

            if self.debug and epoch % 100 == 0:
                res_list = []
                for k in losses[0]:
                    res_list.append(
                        f"{k} {np.mean([loss[k] for loss in losses[-10:]]):.4f}")
                res_str = ", ".join(res_list)
                print(f"    Epoch {epoch: >4}: {res_str}")

        # Return results
        self.sigma = sigma
        return self._predict_internal(inputs, partial_targets)

    def _predict_internal(
        self, inputs: np.ndarray,
        partial_targets: Optional[np.ndarray] = None,
    ) -> SplitResult:

        if self.model is None or self.p_x_y_model is None:
            raise ValueError()

        inference_loader = DataLoader(
            TensorDataset(torch.tensor(inputs, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False,
        )

        # Switch to eval mode
        self.model.eval()
        self.model2.eval()
        self.p_x_y_model.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                logits = 0.5 * (
                    self.model(x_batch)[1] +
                    self.model2(x_batch)[1]
                )
                # pylint: disable=not-callable
                prob = 1 + self.logits_to_evidence(logits)
                prob /= torch.sum(prob, dim=1, keepdim=True)
                all_results.append(prob.cpu().numpy())
            train_probs = np.vstack(all_results)

        if partial_targets is not None:
            train_probs = train_probs * partial_targets

        return SplitResult.from_scores(self.rng, train_probs)

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        return self._predict_internal(inputs)
