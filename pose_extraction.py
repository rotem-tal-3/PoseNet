import cv2
import numpy as np
import os
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd


def print_label_counts(csv_path: str, label_column: str) -> None:
    df = pd.read_csv(csv_path)

    label_counts = df[label_column].value_counts(dropna=False)

    for label, count in label_counts.items():
        print(f"{label}: {count}")


def remove_labels_and_save(input_csv: str,
                           output_csv: str,
                           label_column: str,
                           labels_to_remove) -> None:
    df = pd.read_csv(input_csv)
    filtered_df = df[~df[label_column].isin(labels_to_remove)]
    filtered_df.to_csv(output_csv, index=False)


base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE  # Change to VIDEO or LIVE_STREAM if needed
)

# print_label_counts('pose_home_dataset.csv', "label")
# exit()

def downsample_video_by_time(cap: cv2.VideoCapture, intervals_sec=(0.066, 0.100, 0.180)):
    """
    Given an open cv2.VideoCapture, returns three lists of frames:
    - ~66ms between frames
    - ~100ms between frames
    - ~180ms between frames
    """

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Invalid FPS from VideoCapture")
    # Convert time intervals to frame steps
    frame_steps = [max(1, round(fps * t)) for t in intervals_sec]
    lists = [[] for _ in range(len(intervals_sec))]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for step, frame_list in zip(frame_steps, lists):
            if frame_idx % step == 0:
                frame_list.append(frame)

        frame_idx += 1
    return lists


def get_center_point(landmarks, left_bodypart, right_bodypart):
    left = np.array([landmarks[left_bodypart].x, landmarks[left_bodypart].y])
    right = np.array([landmarks[right_bodypart].x, landmarks[right_bodypart].y])
    return (left + right) / 2


def normalize_pose_landmarks(landmarks):
    # 1. Convert to numpy for easier math
    # We only use x and y for classification (z is often noisy in MediaPipe)
    coords = np.array([[lm.x, lm.y] for lm in landmarks])

    # 2. TRANSLATION: Center around hips
    # Hips are usually landmarks 23 (left) and 24 (right)
    center = (coords[23] + coords[24]) / 2
    coords -= center

    # 3. SCALE: Normalize by torso size
    # Torso size = distance from hip center to shoulder center
    # Shoulders are 11 and 12
    shoulder_center = (coords[11] + coords[12]) / 2
    # We use max distance to ensure we don't divide by zero/noise
    torso_size = np.linalg.norm(shoulder_center) * 2.5  # Scale factor

    # Avoid division by zero
    if torso_size < 1e-6: torso_size = 1.0

    coords /= torso_size

    # Flatten to a single list: [x0, y0, x1, y1, ... x32, y32]
    return coords.flatten().tolist()

mode = 'r'
# MAIN LOOP
with open('pose_home_dataset.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)

    # Add header (label + 66 coordinates)
    header = ['video', 'label']
    for i in range(33):
        header.extend([f'x{i}', f'y{i}'])
    writer.writerow(header)

    dataset_path = "home"
    # 2. Create the detector
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for class_name in os.listdir(dataset_path):
            if class_name.startswith('x_') or class_name.startswith('t_'):
                continue
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path): continue

            print(f"Processing class: {class_name}")

            for video_file in os.listdir(class_path):
                print(video_file)
                cap = cv2.VideoCapture(os.path.join(class_path, video_file))
                downsampled = downsample_video_by_time(cap)

                cons = 0
                for i, sample in enumerate(downsampled):
                    vid_name = f"{video_file}-{i}"
                    no_res = 0
                    for frame in sample:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        # Process Frame
                        results = landmarker.detect(mp_image)
                        if results.pose_landmarks and len(results.pose_landmarks) > 0:
                            if cons > 1: print(cons)
                            cons = 0
                            feature_vector = normalize_pose_landmarks(results.pose_landmarks[0])
                            # Write to CSV
                            writer.writerow([vid_name, class_name] + feature_vector)
                        else:
                            cons += 1
                            no_res += 1
                print(no_res)
                cap.release()

    print("Dataset generation complete!")
