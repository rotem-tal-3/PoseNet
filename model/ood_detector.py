import torch
import numpy as np
from typing import List, Dict, Tuple


class MahalanobisDetector:
    """
    Implements Out-of-Distribution (OOD) detection using the Mahalanobis distance.

    This class models the distribution of feature embeddings for each known class
    as a multivariate Gaussian. During inference, it calculates the distance of
    a new sample to the nearest class centroid. If the distance exceeds a
    calibrated threshold, the sample is flagged as unknown.

    This is generally more robust than Softmax thresholding for detecting
    unseen exercises or random movement.
    """

    def __init__(self, num_classes: int, feature_dim: int):
        self.num_classes = num_classes
        self.mu = torch.zeros(num_classes, feature_dim)
        self.covariance = torch.eye(feature_dim)
        self.precision = torch.eye(feature_dim)
        self.has_fit = False

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Fits the Gaussian parameters (mean and tied covariance) based on training data.

        Args:
            embeddings (torch.Tensor): Tensor of shape (N, Feature_Dim) from the trained model.
            labels (torch.Tensor): Tensor of shape (N,) containing class indices.
        """
        self.mu = self.mu.to(embeddings.device)
        self.covariance = self.covariance.to(embeddings.device)

        for c in range(self.num_classes):
            class_mask = labels == c
            class_embeddings = embeddings[class_mask]

            if class_embeddings.size(0) > 0:
                self.mu[c] = class_embeddings.mean(dim=0)

        centered_data = []
        for c in range(self.num_classes):
            class_mask = labels == c
            if class_mask.sum() > 0:
                centered_data.append(embeddings[class_mask] - self.mu[c])

        if centered_data:
            centered_data = torch.cat(centered_data, dim=0)
            self.covariance = torch.matmul(centered_data.t(), centered_data) / centered_data.size(0)

            reg_lambda = 1e-6
            self.covariance += torch.eye(self.covariance.size(0),
                                         device=embeddings.device) * reg_lambda

            self.precision = torch.inverse(self.covariance)

        self.has_fit = True

    def predict_score(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the minimum Mahalanobis distance for a batch of embeddings.

        Args:
            embedding (torch.Tensor): Tensor of shape (Batch, Feature_Dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - closest_classes: The class index with the minimum distance.
                - distances: The actual Mahalanobis distance values.
        """
        if not self.has_fit:
            raise RuntimeError("Detector must be fit to training data before prediction.")

        batch_size = embedding.size(0)
        embedding = embedding.to(self.mu.device)

        dists = []
        for c in range(self.num_classes):
            delta = embedding - self.mu[c]

            m_dist = torch.diag(torch.matmul(torch.matmul(delta, self.precision), delta.t()))
            dists.append(m_dist.unsqueeze(1))

        dists = torch.cat(dists, dim=1)

        min_dists, closest_classes = torch.min(dists, dim=1)
        return closest_classes, min_dists