import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, List

class StratifiedVideoSplitter:
    """
    Ensures that multiple samples from the same video do not leak across splits.

    This is critical when you have augmented data (e.g., different sampling rates)
    of the same physical performance.

    Args:
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
    """

    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split(self, x: np.ndarray, y: np.ndarray, video_ids: List[str]) -> Tuple[np.ndarray, ...]:
        """
        Splits data into Train, Val, and Test sets based on video groups.

        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): Labels.
            video_ids (List[str]): Unique identifier for the source video of each sample.

        Returns:
            Tuple of (x_train, y_train, x_val, y_val, x_test, y_test).
        """
        gs = GroupShuffleSplit(n_splits=1, train_size=self.train_ratio)
        train_idx, temp_idx = next(gs.split(x, y, groups=video_ids))

        x_train, y_train = x[train_idx], y[train_idx]
        x_temp, y_temp = x[temp_idx], y[temp_idx]
        groups_temp = [video_ids[i] for i in temp_idx]

        test_ratio_adj = 1.0 - (self.val_ratio / (1.0 - self.train_ratio))
        gs_val = GroupShuffleSplit(n_splits=1, train_size=test_ratio_adj)
        val_idx, test_idx = next(gs_val.split(x_temp, y_temp, groups=groups_temp))

        return (
            x_train, y_train,
            x_temp[val_idx], y_temp[val_idx],
            x_temp[test_idx], y_temp[test_idx]
        )


class PoseDataManager:
    """
    Handles datasets with multi-rate sampling to prevent data leakage.

    This ensures that all temporal variations of a specific video performance
    stay within the same data split.
    """

    @staticmethod
    def split_by_video_group(
            x: np.ndarray,
            y: np.ndarray,
            video_ids: List[str],
            train_size: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits features and labels while keeping video groups intact.

        Args:
            x: Feature array (Samples, Channels, Time).
            y: Labels array.
            video_ids: List of source video names (the same for all 3 sampling rates).
            train_size: Proportion of unique videos used for training.

        Returns:
            Tuple: (x_train, x_val, y_train, y_val)
        """
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        train_idx, val_idx = next(gss.split(x, y, groups=video_ids))

        return x[train_idx], x[val_idx], y[train_idx], y[val_idx]

    @staticmethod
    def normalize_keypoints(x: np.ndarray) -> np.ndarray:
        """
        Normalizes MediaPipe landmarks to be relative to the bounding box or center.

        This prevents the model from being biased by the person's
        position in the frame.

        Args:
            x: Raw landmarks (Samples, 66, Time).
        """
        # Example: Center around the midpoint of the hips or a fixed landmark
        # This logic should be identical to what you implement in Kotlin
        mid_hip_idx = 23  # MediaPipe hip landmark
        for s in range(x.shape[0]):
            for t in range(x.shape[2]):
                center_x = x[s, mid_hip_idx, t]
                center_y = x[s, mid_hip_idx + 33, t]
                x[s, :33, t] -= center_x
                x[s, 33:, t] -= center_y
        return x


class PoseNormalizer:
    """
    Normalizes interleaved pose landmarks (x0, y0, x1, y1...) to be invariant
    to subject position and scale.

    Args:
        sequence (torch.Tensor): Shape (Batch, 66, Time) where 66 features
                                 are interleaved [x0, y0, ... x32, y32].
    """

    @staticmethod
    def normalize_sequence(sequence: torch.Tensor) -> torch.Tensor:
        """
        Performs translation to mid-hip and scaling by shoulder width.

        Args:
            sequence: Input tensor of interleaved coordinates.

        Returns:
            torch.Tensor: Normalized interleaved coordinates.
        """
        # Indices for interleaved format
        x_indices = torch.arange(0, 66, 2)
        y_indices = torch.arange(1, 66, 2)

        # 1. Translation: Center on Mid-Hip (Landmarks 23, 24)
        # Landmark 23: x=46, y=47 | Landmark 24: x=48, y=49
        mid_hip_x = (sequence[:, 46, :] + sequence[:, 48, :]) / 2.0
        mid_hip_y = (sequence[:, 47, :] + sequence[:, 49, :]) / 2.0

        normalized = sequence.clone()
        normalized[:, x_indices, :] -= mid_hip_x.unsqueeze(1)
        normalized[:, y_indices, :] -= mid_hip_y.unsqueeze(1)

        # 2. Scaling: Normalize by Euclidean distance between shoulders (Landmarks 11, 12)
        # Landmark 11: x=22, y=23 | Landmark 12: x=24, y=25
        # Calculation: dist = sqrt((x11-x12)^2 + (y11-y12)^2)
        dx = normalized[:, 22, :] - normalized[:, 24, :]
        dy = normalized[:, 23, :] - normalized[:, 25, :]

        shoulder_dist = torch.sqrt(dx ** 2 + dy ** 2).mean(dim=1, keepdim=True)

        epsilon = 1e-6
        scale = shoulder_dist.unsqueeze(1) + epsilon

        return normalized / scale