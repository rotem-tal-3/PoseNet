import torch
import numpy as np
from typing import Tuple
from ood_detector import MahalanobisDetector


class OODCalibrator:
    """
    Handles the inclusion of negative data to find the optimal OOD threshold.
    """

    @staticmethod
    def calibrate_threshold(
            model: torch.nn.Module,
            detector: MahalanobisDetector,
            negative_loader: torch.utils.data.DataLoader,
            percentile: float = 5.0
    ) -> float:
        """
        Calculates a threshold based on 'negative' (non-exercise) data.

        The threshold is set such that the majority of negative data is
        correctly identified as OOD.

        Args:
            model: The trained feature extractor.
            detector: The fitted Mahalanobis detector.
            negative_loader: DataLoader containing "noise" or "non-exercise" poses.
            percentile: The lower-bound percentile of negative scores to use as threshold.

        Returns:
            float: The suggested Mahalanobis distance threshold.
        """
        model.eval()
        negative_scores = []

        with torch.no_grad():
            for inputs, _ in negative_loader:
                embeddings = model.forward_features(inputs)
                _, distances = detector.predict_score(embeddings)
                negative_scores.extend(distances.cpu().numpy().tolist())

        # We want the threshold to be lower than the distances seen in negative data
        return float(np.percentile(negative_scores, percentile))