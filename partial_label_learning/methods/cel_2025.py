""" CEL-2025 by Yang et al.

Code adapted from https://github.com/Yangfc-ML/CEL.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class ClassAssociativeLoss(nn.Module):
    """ CAL """

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, features_w, features_s, labels):
        """ Forward """

        labels_cad_expanded = labels.unsqueeze(1)
        labels_ncad_expanded = (1 - labels).unsqueeze(1)

        mask_cad = labels_cad_expanded * \
            labels_cad_expanded.transpose(1, 2).float()
        mask_ncad = labels_ncad_expanded * \
            labels_ncad_expanded.transpose(1, 2).float()

        mask_pos = mask_cad
        mask_neg = 1 - mask_cad - mask_ncad

        indices = torch.arange(labels.shape[1])
        mask_pos[:, indices, indices] = 0
        mask_neg[:, indices, indices] = 0

        dot_prod_w = torch.matmul(features_w, features_w.transpose(-1, -2))
        dot_prod_s = torch.matmul(features_s, features_s.transpose(-1, -2))

        pos_pairs_mean_w = (mask_pos * dot_prod_w).sum(
            dim=(1, 2)) / (mask_pos.sum(dim=(1, 2)) + 1e-6)
        neg_pairs_mean_w = torch.abs(
            mask_neg * dot_prod_w
        ).sum(dim=(1, 2)) / (mask_neg.sum(dim=(1, 2)) + 1e-6)

        pos_pairs_mean_s = (mask_pos * dot_prod_s).sum(
            dim=(1, 2)) / (mask_pos.sum(dim=(1, 2)) + 1e-6)
        neg_pairs_mean_s = torch.abs(
            mask_neg * dot_prod_s
        ).sum(dim=(1, 2)) / (mask_neg.sum(dim=(1, 2)) + 1e-6)

        loss_w = ((1.0 - pos_pairs_mean_w) +
                  (self.gamma * neg_pairs_mean_w)).mean()
        loss_s = ((1.0 - pos_pairs_mean_s) +
                  (self.gamma * neg_pairs_mean_s)).mean()

        return loss_w + loss_s


class PrototypeDiscriminativeLoss(nn.Module):
    """ PDL """

    def __init__(self, gamma, device):
        super().__init__()
        self.gamma = gamma
        self.device = device

    def forward(self, features_w, features_s, class_wise_prototypes, output, candidate_label):
        """ Forward """

        mask_pos = torch.zeros(
            (output.shape[0], output.shape[1], output.shape[1])).to(self.device)
        mask_neg = torch.zeros(
            (output.shape[0], output.shape[1], output.shape[1])).to(self.device)
        output = torch.where(candidate_label > 0, output, -torch.inf)
        _, index = torch.max(output, 1)

        for i in range(output.shape[0]):
            mask_pos[i, index[i], index[i]] = 1
            mask_neg[i, index[i], :] = 1
            mask_neg[i, index[i], index[i]] = 0

        dot_prod_w = torch.matmul(
            features_w, class_wise_prototypes.transpose(-1, -2))
        dot_prod_s = torch.matmul(
            features_s, class_wise_prototypes.transpose(-1, -2))

        pos_pairs_mean_w = (mask_pos * dot_prod_w).sum(
            dim=(1, 2)) / (mask_pos.sum(dim=(1, 2)) + 1e-6)
        neg_pairs_mean_w = torch.abs(
            mask_neg * dot_prod_w
        ).sum(dim=(1, 2)) / (mask_neg.sum(dim=(1, 2)) + 1e-6)

        pos_pairs_mean_s = (mask_pos * dot_prod_s).sum(
            dim=(1, 2)) / (mask_pos.sum(dim=(1, 2)) + 1e-6)
        neg_pairs_mean_s = torch.abs(
            mask_neg * dot_prod_s
        ).sum(dim=(1, 2)) / (mask_neg.sum(dim=(1, 2)) + 1e-6)

        loss_w = ((1.0 - pos_pairs_mean_w) +
                  (self.gamma * neg_pairs_mean_w)).mean()
        loss_s = ((1.0 - pos_pairs_mean_s) +
                  (self.gamma * neg_pairs_mean_s)).mean()

        return loss_w + loss_s


class LabelDisambiguationLoss(nn.Module):
    """ LDL """

    def __init__(self, predicted_score, predicted_score_weight=0):
        super().__init__()
        self.predicted_score_w = deepcopy(predicted_score)
        self.predicted_score_weight = predicted_score_weight

    def update_predicted_score(self, output_w, index, y):
        """ Update score """

        with torch.no_grad():
            new_w = torch.softmax(output_w, dim=1)

            revised_y = y.clone()
            revised_y_w = revised_y * (new_w + 1e-10)
            revised_y_w = revised_y_w / \
                revised_y_w.sum(dim=1).repeat(
                    revised_y.size(1), 1).transpose(0, 1)

            predicted_score_w = revised_y_w.detach()

            score = self.predicted_score_weight
            self.predicted_score_w[index, :] = score * self.predicted_score_w[
                index, :] + (1 - score) * predicted_score_w

    def forward(self, output_w, output_s, index):
        """ Forward """

        soft_positive_label_w = self.predicted_score_w[
            index, :].clone().detach()

        output_w = F.softmax(output_w, dim=1)
        l_w = soft_positive_label_w * torch.log(output_w + 1e-10)
        loss_w = (-torch.sum(l_w)) / l_w.size(0)

        output_s = F.softmax(output_s, dim=1)
        l_s = soft_positive_label_w * torch.log(output_s + 1e-10)
        loss_s = (-torch.sum(l_s)) / l_s.size(0)

        return loss_w + loss_s


class Cel(PllBaseClassifier):
    """ CEL 2025 by Yang et al. """

    def __init__(self, rng, debug, model, device, **kwargs):
        super().__init__(rng, debug, model, device, **kwargs)
        self.class_wise_prototypes = torch.tensor(0.0, device=self.device)

    def update_prototypes(self, feature_w, logits_w, candidate_label):
        """ Update class prototypes """

        feature_w = feature_w.clone().detach()
        logits_w = logits_w.clone().detach()
        candidate_label = candidate_label.clone().detach()
        logits_w = torch.where(
            candidate_label > 0, logits_w, -torch.inf)

        _, index = torch.max(logits_w, 1)
        for num, feat in enumerate(feature_w):
            self.class_wise_prototypes[index] = (
                self.class_wise_prototypes[index[num]]
                + feat[index[num]]
            )
        self.class_wise_prototypes = F.normalize(
            self.class_wise_prototypes, dim=1)

    def fit(self, inputs, partial_targets):
        """ Fit model """

        # Class-wise embedding
        num_classes = partial_targets.shape[1]
        class_wise_embeddin_model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, num_classes * 32),
        )
        class_wise_embeddin_model.to(self.device)
        class_wise_embeddin_model.compile()
        self.class_wise_prototypes = torch.zeros(
            num_classes, 32, device=self.device)

        # Init loss terms
        train_partial_y_matrix = torch.tensor(
            partial_targets, dtype=torch.float32)
        temp_y = train_partial_y_matrix.sum(dim=1).unsqueeze(
            1).repeat(1, train_partial_y_matrix.shape[1])
        init_confidence = train_partial_y_matrix.float() / temp_y
        init_confidence = init_confidence.to(self.device)
        loss_cls = LabelDisambiguationLoss(
            predicted_score=init_confidence, predicted_score_weight=0)
        loss_cal = ClassAssociativeLoss(gamma=1.0)
        loss_pdl = PrototypeDiscriminativeLoss(gamma=1.0, device=self.device)

        # Data preparation
        x_train = torch.tensor(inputs, dtype=torch.float32)
        y_train = torch.tensor(partial_targets, dtype=torch.float32)
        train_indices = torch.arange(x_train.shape[0], dtype=torch.int32)
        loss_weights = torch.tensor(partial_targets, dtype=torch.float32)
        loss_weights /= loss_weights.sum(dim=1, keepdim=True)
        data_loader = DataLoader(
            TensorDataset(train_indices, x_train, y_train, loss_weights),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        # Optimizer
        self.model.train()
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) +
            list(class_wise_embeddin_model.parameters()),
        )

        # Training loop
        for epoch in self.loop_wrapper(range(self.num_epoch)):
            for idx, inputs_i, partial_targets_i, w_ij in data_loader:
                # Move to device
                inputs_i = inputs_i.to(self.device)
                partial_targets_i = partial_targets_i.to(self.device)
                w_ij = w_ij.to(self.device)

                # Forward pass
                inputs_i_noise = inputs_i + 0.01 * torch.randn_like(inputs_i)
                _, output_w, feature_w = self.model(inputs_i)
                _, output_s, feature_s = self.model(inputs_i_noise)
                feature_w = F.normalize(
                    class_wise_embeddin_model(
                        feature_w).view(-1, num_classes, 32),
                    p=2, dim=2,
                )
                feature_s = F.normalize(
                    class_wise_embeddin_model(
                        feature_s).view(-1, num_classes, 32),
                    p=2, dim=2,
                )
                self.update_prototypes(feature_w, output_w, partial_targets_i)

                # Compute loss
                cls_loss = loss_cls(output_w, output_s, idx)
                ole_loss_sample = loss_cal(
                    feature_w, feature_s, partial_targets_i)
                if epoch <= 500:
                    loss = cls_loss + 0.5 * ole_loss_sample
                else:
                    ole_loss_prototype = loss_pdl(
                        feature_w, feature_s, self.class_wise_prototypes.clone().detach(),
                        output_w, partial_targets_i,
                    )
                    loss = cls_loss + 0.5 * ole_loss_sample + 1.0 * ole_loss_prototype
                loss_cls.update_predicted_score(
                    output_w, idx, partial_targets_i)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Return results
        return self.predict(inputs)

    def predict(self, inputs):
        """ Predict """

        inference_loader = DataLoader(
            TensorDataset(torch.tensor(
                inputs, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False,
        )

        # Switch to eval mode
        self.model.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                all_results.append(
                    self.model(x_batch)[0].cpu().numpy())
            train_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, train_probs)
