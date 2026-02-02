import torch
import json
import numpy as np
from typing import Dict, List, Any
from torch.utils.data import DataLoader, TensorDataset

from net import AdvancedPoseResNet
from ood_detector import MahalanobisDetector
from train import normalize_interleaved_features, prepare_windowed_data


def load_inference_pipeline(model_path: str,
                            ood_params_path: str,
                            num_classes: int,
                            device: torch.device) -> tuple[AdvancedPoseResNet, MahalanobisDetector]:
    """
    Loads the trained model weights and the fitted Mahalanobis detector parameters.

    Args:
        model_path (str): Path to the .pth state dict.
        ood_params_path (str): Path to the .json OOD parameters.
        num_classes (int): Number of exercise classes.
        device (torch.device): Device to load the model onto.

    Returns:
        tuple: (Loaded PoseResNet, Loaded MahalanobisDetector)
    """
    model = AdvancedPoseResNet(input_channels=66, num_classes=num_classes, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_size=device))
    model.to(device)
    model.eval()

    with open(ood_params_path, 'r') as f:
        ood_data = json.load(f)

    detector = MahalanobisDetector(num_classes=num_classes, feature_dim=256)
    detector.mu = torch.tensor(ood_data["mu"]).to(device)
    detector.precision = torch.tensor(ood_data["precision"]).to(device)
    detector.has_fit = True

    return model, detector


def analyze_distances(model: AdvancedPoseResNet,
                      detector: MahalanobisDetector,
                      data_x: np.ndarray,
                      dataset_name: str = "Target",
                      batch_size: int = 64) -> List[float]:
    """
    Runs inference on a dataset and extracts the Mahalanobis distances.

    Args:
        model (AdvancedPoseResNet): The feature extractor.
        detector (MahalanobisDetector): The OOD logic.
        data_x (np.ndarray): Interleaved raw coordinates (N, 66, 10).
        dataset_name (str): Label for printing results.
        batch_size (int): Size of inference batches.

    Returns:
        List[float]: A list of all calculated distances.
    """
    device = next(model.parameters()).device

    # Ensure data is normalized before analysis
    normalized_x = torch.FloatTensor(normalize_interleaved_features(data_x))

    loader = DataLoader(TensorDataset(normalized_x), batch_size=batch_size, shuffle=False)
    all_distances = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            embeddings = model.forward_features(batch_x)
            _, distances = detector.predict_score(embeddings)
            all_distances.extend(distances.cpu().numpy().tolist())

    print(f"\n--- Statistics for {dataset_name} ---")
    print(f"Mean Distance:   {np.mean(all_distances):.4f}")
    print(f"Median Distance: {np.median(all_distances):.4f}")
    print(f"Min Distance:    {np.min(all_distances):.4f}")
    print(f"Max Distance:    {np.max(all_distances):.4f}")
    print(f"95th Percentile: {np.percentile(all_distances, 95):.4f}")
    print(f"99th Percentile: {np.percentile(all_distances, 99):.4f}")

    return all_distances


def run_threshold_finding_session(clean_data: np.ndarray,
                                  random_movement_data: np.ndarray,
                                  model_path: str = "exercise_model_best.pth",
                                  ood_path: str = "exercise_model_ood.json",
                                  num_classes: int = 14):
    """
    Compares distances between clean exercise data and random noise to find
    an optimal threshold for OOD detection.

    Args:
        clean_data (np.ndarray): Known exercise samples (N, 66, 10).
        random_movement_data (np.ndarray): Non-exercise/noise samples (M, 66, 10).
        model_path (str): File path to weights.
        ood_path (str): File path to Mahalanobis JSON.
        num_classes (int): Number of trained classes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, detector = load_inference_pipeline(model_path, ood_path, num_classes, device)

    print("Analyzing In-Distribution (Clean) Data...")
    in_dist_scores = analyze_distances(model, detector, clean_data, "Clean Exercises")

    print("\nAnalyzing Out-of-Distribution (Random) Data...")
    out_dist_scores = analyze_distances(model, detector, random_movement_data, "Random Movement")

    # Suggestion logic
    suggested_min = np.percentile(in_dist_scores, 98)
    suggested_max = np.percentile(out_dist_scores, 2)

    print("\n--- Recommendation ---")
    if suggested_min < suggested_max:
        print(
            f"Clear separation found. Recommended threshold: {(suggested_min + suggested_max) / 2:.4f}")
    else:
        print(f"Distributions overlap. Try a threshold around: {suggested_min:.4f}")
