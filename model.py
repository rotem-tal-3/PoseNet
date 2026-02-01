from keras import layers, models

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from keras.callbacks import ModelCheckpoint
import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tqdm.keras import TqdmCallback
import torch
import torch.nn as nn
import torch.nn.functional as F


WINDOW_SHAPE = 10


class PoseTemporalCNN(nn.Module):
    """
    A 1D CNN designed for temporal sequences of pose landmarks.

    Args:
        num_landmarks (int): Number of landmarks from MediaPipe (default 33).
        input_dims (int): Coordinates per landmark (x, y = 2).
        sequence_length (int): Number of frames in the temporal window (e.g., 10).
        num_classes (int): Number of exercise categories.
        hidden_dim (int): Size of the embedding vector for OOD detection.
    """

    def __init__(self,
                 num_landmarks: int = 33,
                 input_dims: int = 2,
                 sequence_length: int = 10,
                 num_classes: int = 5,
                 hidden_dim: int = 128):
        super(PoseTemporalCNN, self).__init__()

        self.input_channels = num_landmarks * input_dims

        self.conv_block = nn.Sequential(
            nn.Conv1d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_embedding = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.5)
        self.fc_classifier = nn.Linear(hidden_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts temporal embeddings from a window of frames.

        Args:
            x (torch.Tensor): Tensor of shape (Batch, Channels, Sequence_Length).

        Returns:
            torch.Tensor: Embedding of shape (Batch, hidden_dim).
        """
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc_embedding(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.

        Args:
            x (torch.Tensor): Tensor of shape (Batch, Channels, Sequence_Length).

        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes).
        """
        features = self.forward_features(x)
        features = self.dropout(features)
        return self.fc_classifier(features)



def build_pose_model(input_shape=(WINDOW_SHAPE, 66), num_classes=18):
    model = models.Sequential([
        # Layer 1: Learn local movement between adjacent frames
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),

        # Layer 2: Learn complex patterns across the whole second of motion
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.GlobalMaxPooling1D(),  # Condenses the sequence into the 'strongest' signals

        # Layer 3: Reasoning
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # CRITICAL to stop the model from 'memorizing' noise

        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Probability for each exercise
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def prepare_windowed_data(df, window_size=WINDOW_SHAPE, step=1):
    X, y = [], []

    # Process each video individually to avoid 'teleportation' noise
    for video_id, group in df.groupby('video'):
        # Convert landmarks to a numpy array (Shape: Frames x 66)
        features = group.drop(columns=['video', 'label']).values
        label = group['label'].iloc[0]  # Assumes one label per video snippet

        # Create sliding windows
        for i in range(0, len(features) - window_size + 1, step):
            window = features[i: i + window_size]
            X.append(window)
            y.append(label)

    return np.array(X), np.array(y)


def save_from_ckpt(ckpt="best_pose_model.keras"):
    mod = keras.models.load_model(ckpt)
    converter = tf.lite.TFLiteConverter.from_keras_model(mod)

    tflite_model = converter.convert()
    with open('best.tflite', 'wb') as f:
        f.write(tflite_model)


def extract_data_for_vids(data, data_path="reps.csv"):
    df = pd.read_csv('pose_dataset_clean.csv')
    vids, reps = zip(*data)
    mask = df['video'].str.startswith(tuple(vids), na=False)
    filtered_df = df[mask].copy()

    def find_value(val):
        for prefix, target_val in data:
            if val.startswith(prefix):
                return target_val
        return None

    filtered_df['reps'] = filtered_df['video'].apply(find_value)
    filtered_df.to_csv("reps.csv")

data = [("stu2_29", 27), ("6-45", 33), ("barbell biceps curl_11", 3), ("barbell biceps curl_19",4),
        ("stu5_33.mp", 8), ("stu2_41.mp4-4-48", 23), ("stu2_47", 20), ("stu3_50.mp4-1-101", 29),
        ("17-57", 16), ("stu6_4", 5), ("stu2_55", 30), ("stu4_53", 10), ("stu2_56", 20),
        ("stu1_67.mp4-4-41", 20)]
extract_data_for_vids(data)
# save_from_ckpt()

def train_model():

    v_prob = """v_Lunges_g01_c05.avi
    v_Lunges_g04_c02.avi
    stu1_63.mp4
    stu8_58.mp4-0-9.mp4
    stu1_72.mp4
    Incline_Dumbbell_Bench_Press_-_OPEX_Exercise_Library_240P.mp4
    stu1_11.mp4
    stu1_4.mp4
    stu5_6.mp4
    v_Lunges_g01_c03.avi
    v_Lunges_g01_c05.avi
    v_Lunges_g19_c07.avi
    31.0-73.0.mp4
    Bent_Over_Reverse_Dumbbell_Fly_240P.mp4
    How_to_do_a_perfect_dumbbell_bent_over_fly_240P.mp4
    stu1_61.mp4
    """.split('\n')


    problematic = """v_Lunges_g01_c05.avi
    v_Lunges_g04_c02.avi
    v_Lunges_g07_c03.avi
    v_Lunges_g19_c07.avi
    v_Lunges_g21_c03.avi
    stu3_36.mp4-0-10.mp4
    stu4_32.mp4-0-25.mp4
    stu5_33.mp4-0-45.mp4
    v_PullUps_g07_c04.avi
    v_PullUps_g09_c01.avi
    stu2_45.mp4
    stu4_44.mp4-5-38.mp4
    v_PushUps_g07_c01.avi
    stu1_63.mp4
    stu8_58.mp4-0-9.mp4
    stu5_6.mp4
    stu1_72.mp4""".split('\n')

    # 1. Load your consolidated CSV
    df = pd.read_csv('pose_dataset_clean.csv')
    for p in v_prob:
        df = df[~(df["video"] == p)]
    print(df[df.isna().any(axis=1)])

    # 2. Split by Video ID (80% Train, 20% Test)
    # This prevents data leakage!
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['video']))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train, y_train_raw = prepare_windowed_data(train_df)
    X_test, y_test_raw = prepare_windowed_data(test_df)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_raw)
    y_test_encoded = encoder.fit_transform(y_test_raw)
    num_classes = len(encoder.classes_)
    y_train = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test_encoded, num_classes)

    model = build_pose_model(num_classes=num_classes)

    callbacks = [ModelCheckpoint('best_pose_model.keras', save_best_only=True, monitor='val_accuracy',
                                 mode='max'),
                 ModelCheckpoint('latest_pose_model.keras', save_best_only=False),
                 TqdmCallback(verbose=1)]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=callbacks
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)