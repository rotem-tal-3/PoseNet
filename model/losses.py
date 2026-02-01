import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    """
    Combines Class Weighting with Focal Loss to handle extreme class imbalance.

    Focal Loss down-weights easy samples and focuses the model on hard-to-classify
    minority classes. Class weights explicitly counteract the frequency bias.

    Args:
        alpha (torch.Tensor): A tensor of weights for each class.
        gamma (float): Focusing parameter. Higher values reduce loss for easy samples.
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the focal loss between input logits and target labels.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def calculate_class_weights(counts) -> torch.Tensor:
    """
    Calculates inverse frequency weights to balance the loss function.

    Args:
        counts (List[int]): Number of samples per class in order of class index.

    Returns:
        torch.Tensor: Weights tensor.
    """
    total = sum(counts)
    weights = [total / (len(counts) * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)