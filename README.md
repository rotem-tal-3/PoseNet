# Exercise Recognition with MediaPipe Pose & Mahalanobis OOD Detection

This project implements an exercise recognition system based on MediaPipe Pose landmarks and a temporal convolutional neural network trained in PyTorch.
Instead of a standard softmax classifier, the model produces embeddings that are evaluated using a Mahalanobis distance–based detector to handle unknown or out-of-distribution (OOD) exercises.

The trained network can be exported to ONNX and TensorFlow Lite (TFLite) for deployment on edge or mobile devices.

## Overview

Pipeline summary:

- Pose extraction using MediaPipe Pose

- Sliding window temporal representation of landmarks

- 1D CNN–based embedding network with Squeeze-and-Excitation

- Mahalanobis distance–based classification & unknown detection

- Model export to ONNX → TFLite

### Input Representation

Each input sample is a window of pose landmarks over time.

Shape: (1, 66, 10)

66: Pose features (33 landmarks × x,y coordinates)

10: Temporal window size (frames)

1: Batch dimension

Landmarks are assumed to be pre-normalized (e.g. relative to body center / scale).

### Model Architecture

The network is designed for lightweight temporal modeling and edge deployment.

Core Components

- 1D Convolutional Layers
- Temporal convolutions over pose features
- Capture short-term motion patterns
- Squeeze-and-Excitation (SE) Block
- Embedding Layer

Outputs a fixed-length latent representation along with class probabilities.

The latent representation is used for distance-based classification instead of softmax

### Classification & Unknown Detection

Instead of a standard classifier, this project uses a Mahalanobis Detector:

- Extract embeddings for all known exercise classes
- Compute class-wise centroids and covariance matrix
  
During inference:

- Compute embedding for an input window
- Measure Mahalanobis distance to known class centroids
- Assign closest class or label as unknown if distance exceeds a threshold

This approach enables:

- Open-set recognition
- Robust rejection of unseen exercises
- Better generalization with limited labeled data
