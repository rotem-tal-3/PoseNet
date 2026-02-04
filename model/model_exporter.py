import torch
import numpy as np
import json
from typing import Any

from onnx_tf.backend import prepare

from net import AdvancedPoseResNet
from ood_detector import MahalanobisDetector


class ModelExporter:
    """
    Handles the conversion of the PoseCNN model to TFLite and the serialization
    of OOD detector parameters for use in mobile environments.
    """

    @staticmethod
    def export_to_tflite(model: AdvancedPoseResNet,
                         output_path: str,
                         input_shape: tuple = (1, 66, 30)):
        """
        Converts the PyTorch model to TFLite format via ONNX.

        Args:
            model (PoseCNN): The trained PyTorch model.
            output_path (str): File path to save the .tflite model.
            input_shape (tuple): The expected input shape (Batch, Channels, SeqLen).
        """
        import onnx
        import tensorflow as tf

        model.eval()
        dummy_input = torch.randn(*input_shape)
        onnx_path = output_path.replace(".tflite", ".onnx")

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['logits', 'embeddings'],
            dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'},'embeddings': {0: 'batch_size'}},
            opset_version=12
        )

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(output_path.replace(".tflite", "_tf"))

        converter = tf.lite.TFLiteConverter.from_saved_model(output_path.replace(".tflite", "_tf"))
        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

    @staticmethod
    def export_ood_parameters(detector: MahalanobisDetector, output_path: str):
        """
        Exports Mahalanobis parameters (means and precision matrix) to a JSON file.

        Args:
            detector (MahalanobisDetector): The fitted OOD detector.
            output_path (str): File path to save the parameters.
        """
        data = {
            "mu": detector.mu.cpu().numpy().tolist(),
            "precision": detector.precision.cpu().numpy().tolist(),
            "num_classes": detector.num_classes
        }
        with open(output_path, "w") as f:
            json.dump(data, f)

def run_export():
    """
    Example script to export the temporal model for Android.
    """
    config = {
        "num_landmarks": 33,
        "input_dims": 2,
        "sequence_length": 10,
        "num_classes": 5,
        "hidden_dim": 128
    }

    model = AdvancedPoseResNet(**config)
    # Load your trained weights here: model.load_state_dict(torch.load('path.pth'))

    exporter = ModelExporter()

    # Input shape for Conv1d: (Batch, Channels, Time)
    input_shape = (1, config["num_landmarks"] * config["input_dims"], config["sequence_length"])

    exporter.export_to_tflite(
        model=model,
        output_path="exercise_classifier.tflite",
        input_shape=input_shape
    )
