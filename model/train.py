import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tensorflow
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import json
import pandas as pd

from net import AdvancedPoseResNet
from ood_detector import MahalanobisDetector
from model_exporter import ModelExporter


def prepare_windowed_data(df, window_size=10, step=1):
    X, y, vid  = [], [], []
    mapping = dict()
    lab_ind = 0
    highest = 0
    # Process each video individually to avoid 'teleportation' noise
    for video_id, group in df.groupby('video'):
        # Convert landmarks to a numpy array (Shape: Frames x 66)
        features = group.drop(columns=['video', 'label']).values
        label = group['label'].iloc[0]  # Assumes one label per video snippet
        if label in mapping.keys():
            lab_ind = mapping[label]
        else:
            mapping[label] = highest
            lab_ind = highest
            highest += 1

        # Create sliding windows
        for i in range(0, len(features) - window_size + 1, step):
            window = features[i: i + window_size]
            X.append(window)
            y.append(lab_ind)
            vid.append(video_id)

    return np.array(X), np.array(y), np.array(vid), mapping


class WeightedFocalLoss(nn.Module):
    """
    Implementation of Focal Loss with class-wise weighting.

    Args:
        alpha (torch.Tensor): Weights for each class.
        gamma (float): Focusing parameter to prioritize hard samples.
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def normalize_interleaved_features(features: np.ndarray) -> np.ndarray:
    """
    Normalizes interleaved keypoints (x0, y0, x1, y1...) across batch and time.

    Args:
        features (np.ndarray): Shape (N, 66, 10).

    Returns:
        np.ndarray: Normalized features.
    """
    normalized = features.copy()
    x_idxs = np.arange(0, 66, 2)
    y_idxs = np.arange(1, 66, 2)

    for i in range(features.shape[0]):
        for t in range(features.shape[2]):
            frame = features[i, :, t]

            mid_hip_x = (frame[46] + frame[48]) / 2.0
            mid_hip_y = (frame[47] + frame[49]) / 2.0

            dx = (frame[22] - mid_hip_x) - (frame[24] - mid_hip_x)
            dy = (frame[23] - mid_hip_y) - (frame[25] - mid_hip_y)
            shoulder_dist = np.sqrt(dx ** 2 + dy ** 2)

            scale = shoulder_dist if shoulder_dist > 1e-6 else 1.0

            normalized[i, x_idxs, t] = (features[i, x_idxs, t] - mid_hip_x) / scale
            normalized[i, y_idxs, t] = (features[i, y_idxs, t] - mid_hip_y) / scale

    return normalized


def train_pose_classifier(features: np.ndarray,
                          labels: np.ndarray,
                          video_ids: np.ndarray,
                          output_prefix: str = "exercise_model",
                          epochs: int = 20,
                          batch_size: int = 64,
                          learning_rate: float = 1e-3):
    """
    Orchestrates the full training pipeline: normalization, video-aware split,
    weighted training, OOD calibration, and TFLite export.

    Args:
        features (np.ndarray): Raw interleaved features (N, 66, 10).
        labels (np.ndarray): Integer labels (N,).
        video_ids (np.ndarray): Identifiers for source videos (N,).
        output_prefix (str): Base name for exported files.
        epochs (int): Training iterations.
        batch_size (int): Samples per batch.
        learning_rate (float): Initial optimizer step size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(labels))

    normalized_x = normalize_interleaved_features(features)

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(normalized_x, labels, groups=video_ids))

    x_train = torch.FloatTensor(normalized_x[train_idx])
    y_train = torch.LongTensor(labels[train_idx])
    x_val = torch.FloatTensor(normalized_x[val_idx])
    y_val = torch.LongTensor(labels[val_idx])

    _, counts = np.unique(labels[train_idx], return_counts=True)
    weights = torch.FloatTensor(sum(counts) / (len(counts) * counts)).to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)

    model = AdvancedPoseResNet(input_channels=66, num_classes=num_classes, hidden_dim=256).to(
        device)
    criterion = WeightedFocalLoss(alpha=weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs[0], batch_y).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch: {epoch + 1} | Loss: {avg_val_loss:.4f} Train Loss: {train_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{output_prefix}_best.pth")

    model.load_state_dict(torch.load(f"{output_prefix}_best.pth"))

    all_embeddings = []
    all_train_labels = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            all_embeddings.append(model.forward_features(batch_x.to(device)))
            all_train_labels.append(batch_y)

    detector = MahalanobisDetector(num_classes=num_classes, feature_dim=256)
    detector.fit(torch.cat(all_embeddings), torch.cat(all_train_labels).to(device))

    exporter = ModelExporter()
    exporter.export_to_tflite(model, f"{output_prefix}.tflite", input_shape=(1, 66, 10))
    exporter.export_ood_parameters(detector, f"{output_prefix}_ood.json")

    with open(f"{output_prefix}_metadata.json", "w") as f:
        json.dump({
            "num_classes": num_classes,
            "window_size": 10,
            "feature_size": 66,
            "normalization": "interleaved_mid_hip_shoulder_scaled"
        }, f)


def remove_labels_and_save(input_csv: str,
                           label_column: str,
                           labels_to_remove) -> None:
    df = pd.read_csv(input_csv)
    filtered_df = df[~df[label_column].isin(labels_to_remove)]
    return filtered_df
