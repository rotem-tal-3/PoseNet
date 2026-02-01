import torch
from torch.utils.data import DataLoader
from models import PoseCNN
from ood_detector import MahalanobisDetector


def fit_ood_detector(model: PoseCNN,
                     train_loader: DataLoader,
                     num_classes: int,
                     feature_dim: int,
                     device: str = 'cpu') -> MahalanobisDetector:
    """
    Utility function to run a pass over the training data and fit the OOD detector.

    This should be called after the PoseCNN has finished training.

    Args:
        model (PoseCNN): Trained model instance.
        train_loader (DataLoader): Loader containing the training dataset.
        num_classes (int): Number of classes.
        feature_dim (int): Dimension of the embedding layer in PoseCNN.
        device (str): Computation device.

    Returns:
        MahalanobisDetector: The fitted detector ready for inference.
    """
    model.eval()
    model.to(device)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            embeddings = model.forward_features(inputs)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    detector = MahalanobisDetector(num_classes=num_classes, feature_dim=feature_dim)
    detector.fit(all_embeddings, all_labels)

    return detector