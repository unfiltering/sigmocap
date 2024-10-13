import cv2
import numpy as np
import mediapipe as mp
import json
import os
import time

# Initialize Mediapipe holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to load recorded data
def load_recorded_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to draw skeleton
def draw_skeleton(image, joints, bone_color, show_lines):
    pose_connections = [
        (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_ELBOW),
        (mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.LEFT_WRIST),
        (mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_ELBOW),
        (mp_holistic.PoseLandmark.RIGHT_ELBOW, mp_holistic.PoseLandmark.RIGHT_WRIST),
        (mp_holistic.PoseLandmark.LEFT_HIP, mp_holistic.PoseLandmark.LEFT_KNEE),
        (mp_holistic.PoseLandmark.LEFT_KNEE, mp_holistic.PoseLandmark.LEFT_ANKLE),
        (mp_holistic.PoseLandmark.RIGHT_HIP, mp_holistic.PoseLandmark.RIGHT_KNEE),
        (mp_holistic.PoseLandmark.RIGHT_KNEE, mp_holistic.PoseLandmark.RIGHT_ANKLE),
        (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_SHOULDER),
        (mp_holistic.PoseLandmark.LEFT_HIP, mp_holistic.PoseLandmark.RIGHT_HIP),
        (mp_holistic.PoseLandmark.LEFT_ANKLE, mp_holistic.PoseLandmark.LEFT_FOOT_INDEX),
        (mp_holistic.PoseLandmark.RIGHT_ANKLE, mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX),
    ]

    if show_lines:
        for start_idx, end_idx in pose_connections:
            start = joints[start_idx.value]
            end = joints[end_idx.value]

            cv2.line(image, tuple((start * image.shape[1::-1]).astype(int)), 
                     tuple((end * image.shape[1::-1]).astype(int)), bone_color, 2)

    # Draw individual joints
    for idx, joint in enumerate(joints):
        cv2.circle(image, tuple((joint * image.shape[1::-1]).astype(int)), 5, bone_color, -1)

def main():
    # Set video capture for display (can also be set to 0 for webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    bone_color = (255, 255, 255)
    show_lines = True

    # Get the file path to the recorded data
    file_path = input("Enter the path to the recorded motion data (.sr file): ")
    
    # Load recorded data
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    recorded_data = load_recorded_data(file_path)
    total_frames = len(recorded_data)

    # Start the replay
    for frame_idx in range(total_frames):
        frame_data = recorded_data[frame_idx]

        # Create a blank frame to draw the skeleton
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Prepare joints array
        joints = np.array([frame_data[f'joint_{i}'] for i in range(33)], dtype=np.float32)

        # Draw the skeleton on the frame
        draw_skeleton(frame, joints, bone_color, show_lines)

        # Display the frame
        cv2.imshow("Replay SigMocap", frame)

        # Frame rate control (optional)
        time.sleep(0.033)  # ~30 FPS

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
