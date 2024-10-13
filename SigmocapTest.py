import cv2
import numpy as np
import mediapipe as mp
import time
import os
import json
from datetime import datetime

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

def lerp(start, end, t):
    return start + (end - start) * t

def average_positions(positions):
    valid_positions = [pos for pos in positions if pos is not None]
    return np.mean(valid_positions, axis=0) if valid_positions else None

def draw_skeleton(image, pose_landmarks, hand_landmarks, bone_color, last_positions, show_lines):
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
    
    joints = np.array([(landmark.x, landmark.y) for landmark in pose_landmarks.landmark], dtype=np.float32)

    if show_lines:
        for start_idx, end_idx in pose_connections:
            start = joints[start_idx.value]
            end = joints[end_idx.value]

            if start[0] > 0 and start[1] > 0:
                last_positions[start_idx.value] = start
            if end[0] > 0 and end[1] > 0:
                last_positions[end_idx.value] = end

            cv2.line(image, tuple((start * image.shape[1::-1]).astype(int)), tuple((end * image.shape[1::-1]).astype(int)), bone_color, 2)

    left_hip, right_hip = joints[mp_holistic.PoseLandmark.LEFT_HIP.value], joints[mp_holistic.PoseLandmark.RIGHT_HIP.value]
    left_shoulder, right_shoulder = joints[mp_holistic.PoseLandmark.LEFT_SHOULDER.value], joints[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

    mid_hip = (left_hip + right_hip) / 2
    neck = (left_shoulder + right_shoulder) / 2
    top_of_head = np.array([neck[0], neck[1] - 0.15])

    if show_lines:
        cv2.line(image, tuple((mid_hip * image.shape[1::-1]).astype(int)), tuple((neck * image.shape[1::-1]).astype(int)), bone_color, 2)
        cv2.line(image, tuple((neck * image.shape[1::-1]).astype(int)), tuple((top_of_head * image.shape[1::-1]).astype(int)), bone_color, 2)

    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmark,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=bone_color, thickness=2, circle_radius=0),
                mp_drawing.DrawingSpec(color=bone_color, thickness=2)
            )

def calculate_brightness(frame):
    return np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

# Set up recording variables
recording = False
recorded_data = []

historical_positions = {i: [] for i in range(33)}
last_smoothed_positions = {i: None for i in range(33)}
smoothing_duration = 0.5
frame_time = time.time()

show_camera = True  # Set to True to show the camera display by default
show_lines = True
bone_color = (255, 255, 255)

while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        average_brightness = calculate_brightness(frame)
        holistic.min_detection_confidence = 0.6 if average_brightness < 100 else 0.4

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        holistic_results = holistic.process(image_rgb)

        # Create a blank frame to draw on
        mocap_frame = np.zeros_like(frame)

        if holistic_results.pose_landmarks:
            joints = np.array([(landmark.x, landmark.y) for landmark in holistic_results.pose_landmarks.landmark], dtype=np.float32)

            # Record positions if recording is active
            if recording:
                bone_data = {}
                for idx, joint in enumerate(joints):
                    bone_data[f'joint_{idx}'] = joint.tolist()  # Store positions as lists for JSON serialization
                recorded_data.append(bone_data)

            for idx, joint in enumerate(joints):
                historical_positions[idx].append(joint)
                if len(historical_positions[idx]) > int(smoothing_duration / (1 / 30)):
                    historical_positions[idx].pop(0)

                if time.time() - frame_time >= smoothing_duration:
                    averaged_position = average_positions(historical_positions[idx])
                    if averaged_position is not None:
                        last_smoothed_positions[idx] = lerp(
                            last_smoothed_positions[idx] if last_smoothed_positions[idx] is not None else joint,
                            averaged_position,
                            0.1
                        )
                    historical_positions[idx] = []

            frame_time = time.time()

        # Always draw the skeleton on the mocap frame
        draw_skeleton(mocap_frame, holistic_results.pose_landmarks, [holistic_results.left_hand_landmarks, holistic_results.right_hand_landmarks], bone_color, last_smoothed_positions, show_lines)

        # Combine the original frame and the mocap frame
        if show_camera:
            display_frame = cv2.addWeighted(frame, 0.5, mocap_frame, 0.5, 0)
        else:
            display_frame = mocap_frame

        cv2.imshow("SigMocap Debug", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            show_camera = not show_camera  # Toggle camera display
        elif key == ord('l'):
            show_lines = not show_lines
        elif key == ord('r'):  # Press 'r' to start/stop recording
            recording = not recording
            if recording:
                print("Recording started.")
                recorded_data = []  # Reset recorded data when starting new recording
            else:
                print("Recording stopped.")
                # Save the recorded data to a file
                date_str = datetime.now().strftime("%m-%d-%Y")
                directory = os.path.join(os.path.expanduser("~"), "AppData", "Local", "SigMocap")
                os.makedirs(directory, exist_ok=True)
                file_path = os.path.join(directory, f"{date_str}.sr")
                with open(file_path, 'w') as f:
                    json.dump(recorded_data, f)
                print(f"Data saved to {file_path}.")

    except Exception as e:
        print(f"Error: {e}")

cap.release()
cv2.destroyAllWindows()
