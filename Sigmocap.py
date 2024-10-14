import os
import cv2
import numpy as np
import mediapipe as mp
import json
from datetime import datetime
import time
import logging
import base64
import requests
import socket  # For getting the hostname

# Setup logging
appdata_dir = os.path.expanduser(r'~/AppData/Local/SigMocap/')
logs_dir = os.path.join(appdata_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)  # Ensure the logs directory exists

log_file_path = os.path.join(logs_dir, 'sigmocap.log')

# Check if log file exists; if not, create it
if not os.path.exists(log_file_path):
    open(log_file_path, 'w').close()

# Generate HWID based on hostname and current time
def get_hwid():
    hostname = socket.gethostname()
    hwid_str = f"{hostname}"
    # Encode the HWID string in Base64
    hwid_encoded = base64.b64encode(hwid_str.encode()).decode()
    return hwid_encoded

hwid = get_hwid()

logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,
    format=f'[SigMocap %(levelname)s]: %(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)

logging.info("SigMocap init.")

# Custom encryption and decryption functions
def custom_encrypt(data: bytes, shift: int = 5) -> str:
    """Encrypt the given data by shifting characters by a fixed amount."""
    encrypted = ''.join(chr((b + shift) % 256) for b in data)
    return encrypted

def custom_decrypt(encrypted_data: str, shift: int = 5) -> bytes:
    """Decrypt the given data by shifting characters back by the fixed amount."""
    decrypted = bytes((ord(char) - shift) % 256 for char in encrypted_data)
    return decrypted

# Function to send files to Discord
def send_to_discord(webhook_url, files, hwid):
    """Send log and replay files to Discord webhook with HWID."""
    for index, file_path in enumerate(files, start=1):
        # Prepare the message content
        message_content = f""
        if index == 1:
            message_content = f"# Telemetry\nIdentifier: {hwid}"

        # Send the file to Discord
        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    webhook_url,
                    data={"content": message_content},  # Use 'data' for content
                    files={"file": f},  # Attach the file
                )

        except Exception as e:
            print(f"")

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic_model = mp_holistic.Holistic(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Capture video from the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Error: Could not open video capture.")
    exit()

# Function to linear interpolate for smoothing
def lerp(start, end, t):
    return start + (end - start) * t

# Function to average positions for smoothing
def average_positions(positions):
    valid_positions = [pos for pos in positions if pos is not None]
    return np.mean(valid_positions, axis=0) if valid_positions else None

# Function to draw the skeleton and hand landmarks
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

    # Get joint coordinates
    joints = np.array(
        [(landmark.x, landmark.y) for landmark in pose_landmarks.landmark],
        dtype=np.float32,
    )

    # Draw pose connections and hand landmarks
    if show_lines:
        for start_idx, end_idx in pose_connections:
            start, end = joints[start_idx.value], joints[end_idx.value]
            if np.any(start > 0) and np.any(end > 0):
                last_positions[start_idx.value], last_positions[end_idx.value] = (
                    start,
                    end,
                )
                cv2.line(
                    image,
                    tuple((start * image.shape[1::-1]).astype(int)),
                    tuple((end * image.shape[1::-1]).astype(int)),
                    bone_color,
                    2,
                )

        # Draw centerline (spine)
        mid_hip = (
            joints[mp_holistic.PoseLandmark.LEFT_HIP.value]
            + joints[mp_holistic.PoseLandmark.RIGHT_HIP.value]
        ) / 2
        neck = (
            joints[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
            + joints[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        ) / 2
        top_of_head = np.array([neck[0], neck[1] - 0.15])

        cv2.line(
            image,
            tuple((mid_hip * image.shape[1::-1]).astype(int)),
            tuple((neck * image.shape[1::-1]).astype(int)),
            bone_color,
            2,
        )
        cv2.line(
            image,
            tuple((neck * image.shape[1::-1]).astype(int)),
            tuple((top_of_head * image.shape[1::-1]).astype(int)),
            bone_color,
            2,
        )

    # Draw hand landmarks
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            if hand_landmark:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmark,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=bone_color, thickness=2, circle_radius=0
                    ),
                    mp_drawing.DrawingSpec(color=bone_color, thickness=2),
                )

# Function to calculate frame brightness
def calculate_brightness(frame):
    return np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

# Initialize variables
bin = True
recorded_data = []
historical_positions = {i: [] for i in range(33)}
last_smoothed_positions = {i: None for i in range(33)}
smoothing_duration = 0.3  # Shorter duration for faster smoothing

frame_time = time.time()
screen_width, screen_height = 1920, 1080  # Customize based on display

# Set up window
cv2.namedWindow("SigMocap", cv2.WINDOW_NORMAL)

# Function to extract and structure frame data for recording
def extract_frame_data(joints, left_hand_landmarks, right_hand_landmarks):
    bone_mapping = {
        0: "Nose",
        1: "Left_Eye",
        2: "Right_Eye",
        3: "Left_Ear",
        4: "Right_Ear",
        5: "Left_Shoulder",
        6: "Right_Shoulder",
        7: "Left_Elbow",
        8: "Right_Elbow",
        9: "Left_Wrist",
        10: "Right_Wrist",
        11: "Left_Hip",
        12: "Right_Hip",
        13: "Left_Knee",
        14: "Right_Knee",
        15: "Left_Ankle",
        16: "Right_Ankle",
        17: "Left_Foot_Index",
        18: "Right_Foot_Index",
        19: "Left_Thumb_CMC",
        20: "Left_Thumb_MCP",
        21: "Left_Thumb_IP",
        22: "Left_Thumb_Tip",
        23: "Left_Index_Finger_CMC",
        24: "Left_Index_Finger_MCP",
        25: "Left_Index_Finger_PIP",
        26: "Left_Index_Finger_DIP",
        27: "Left_Index_Finger_Tip",
        28: "Left_Middle_Finger_CMC",
        29: "Left_Middle_Finger_MCP",
        30: "Left_Middle_Finger_PIP",
        31: "Left_Middle_Finger_DIP",
        32: "Left_Middle_Finger_Tip",
        33: "Left_Ring_Finger_CMC",
        34: "Left_Ring_Finger_MCP",
        35: "Left_Ring_Finger_PIP",
        36: "Left_Ring_Finger_DIP",
        37: "Left_Ring_Finger_Tip",
        38: "Left_Pinky_Finger_CMC",
        39: "Left_Pinky_Finger_MCP",
        40: "Left_Pinky_Finger_PIP",
        41: "Left_Pinky_Finger_DIP",
        42: "Left_Pinky_Finger_Tip",
        43: "Right_Thumb_CMC",
        44: "Right_Thumb_MCP",
        45: "Right_Thumb_IP",
        46: "Right_Thumb_Tip",
        47: "Right_Index_Finger_CMC",
        48: "Right_Index_Finger_MCP",
        49: "Right_Index_Finger_PIP",
        50: "Right_Index_Finger_DIP",
        51: "Right_Index_Finger_Tip",
        52: "Right_Middle_Finger_CMC",
        53: "Right_Middle_Finger_MCP",
        54: "Right_Middle_Finger_PIP",
        55: "Right_Middle_Finger_DIP",
        56: "Right_Middle_Finger_Tip",
        57: "Right_Ring_Finger_CMC",
        58: "Right_Ring_Finger_MCP",
        59: "Right_Ring_Finger_PIP",
        60: "Right_Ring_Finger_DIP",
        61: "Right_Ring_Finger_Tip",
        62: "Right_Pinky_Finger_CMC",
        63: "Right_Pinky_Finger_MCP",
        64: "Right_Pinky_Finger_PIP",
        65: "Right_Pinky_Finger_DIP",
        66: "Right_Pinky_Finger_Tip",
    }

    frame_data = {
        'joints': {
            bone_mapping[idx]: {
                'x': float(joint[0]),  # Convert to float
                'y': float(joint[1]),  # Convert to float
                'z': float(joints[idx][2]) if len(joints[idx]) > 2 else None  # Convert to float
            }
            for idx, joint in enumerate(joints)
        },
        'left_hand': {},
        'right_hand': {}
    }

    if left_hand_landmarks:
        frame_data['left_hand'] = {
            f'left_hand_joint_{idx}': {
                'x': float(landmark.x),  # Convert to float
                'y': float(landmark.y),  # Convert to float
                'z': float(landmark.z) if hasattr(landmark, 'z') else None  # Convert to float
            } for idx, landmark in enumerate(left_hand_landmarks.landmark)
        }
    else:
        frame_data['left_hand'] = None  # Explicitly set to None if not detected

    if right_hand_landmarks:
        frame_data['right_hand'] = {
            f'right_hand_joint_{idx}': {
                'x': float(landmark.x),  # Convert to float
                'y': float(landmark.y),  # Convert to float
                'z': float(landmark.z) if hasattr(landmark, 'z') else None  # Convert to float
            } for idx, landmark in enumerate(right_hand_landmarks.landmark)
        }
    else:
        frame_data['right_hand'] = None  # Explicitly set to None if not detected

    return frame_data

# Function to save recorded data
def save_recording(recorded_data):
    date_str = datetime.now().strftime("%H-%M.%m-%d-%Y")
    directory = os.path.join(
        os.path.expanduser("~"), "AppData", "Local", "SigMocap", "bin"
    )
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{date_str}.sr")

    # Encrypt the recorded data before saving
    json_data = json.dumps(recorded_data)  # Convert to JSON string
    # Encode in base64 before custom encrypting
    base64_data = base64.b64encode(json_data.encode()).decode()
    encrypted_data = custom_encrypt(base64_data.encode())  # Encrypt the Base64 data

    with open(file_path, "w") as f:
        f.write(encrypted_data)  # Write the encrypted data to file
    logging.info(f"Data saved to {file_path}.")
    return file_path  # Return the file path for further use

# Main loop
while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Failed to capture image.")
            break
        
        # Resize frame based on screen size while maintaining aspect ratio
        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_width = (
            screen_width
            if screen_width / screen_height <= aspect_ratio
            else int(screen_height * aspect_ratio)
        )
        new_height = int(new_width / aspect_ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Create a black background and center the resized frame
        display_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        y_offset = (screen_height - new_height) // 2
        display_frame[
            y_offset : y_offset + new_height,
            (screen_width - new_width) // 2 : (screen_width + new_width) // 2,
        ] = resized_frame

        # Adjust holistic model based on frame brightness
        holistic_model.min_detection_confidence = (
            0.7 if calculate_brightness(display_frame) < 100 else 0.4
        )

        # Process the frame
        image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        holistic_results = holistic_model.process(image_rgb)

        # Prepare a blank frame for drawing mocap data
        mocap_frame = np.zeros_like(display_frame)

        if holistic_results.pose_landmarks:
            joints = np.array(
                [
                    (landmark.x, landmark.y)
                    for landmark in holistic_results.pose_landmarks.landmark
                ],
                dtype=np.float32,
            )

            # Update the recording regardless of whether we are actively tracking
            if bin:
                frame_data = extract_frame_data(
                    joints,
                    holistic_results.left_hand_landmarks,
                    holistic_results.right_hand_landmarks,
                )
                recorded_data.append(frame_data)

            # Update historical positions and smooth
            for idx, joint in enumerate(joints):
                historical_positions[idx].append(joint)
                if len(historical_positions[idx]) > int(smoothing_duration / (1 / 30)):
                    historical_positions[idx].pop(0)

                if time.time() - frame_time >= smoothing_duration:
                    averaged_position = average_positions(historical_positions[idx])
                    if averaged_position is not None:
                        last_smoothed_positions[idx] = lerp(
                            last_smoothed_positions[idx]
                            if last_smoothed_positions[idx] is not None
                            else joint,
                            averaged_position,
                            0.1,
                        )
                    historical_positions[idx] = []

            frame_time = time.time()

        # Draw skeleton and hand data
        draw_skeleton(
            display_frame,
            holistic_results.pose_landmarks,
            [
                holistic_results.left_hand_landmarks,
                holistic_results.right_hand_landmarks,
            ],
            (255, 255, 255),
            last_smoothed_positions,
            True,
        )

        # Display the combined frame
        cv2.imshow("SigMocap", display_frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logging.info("Exiting program.")
            break
        
        # Automatically save recording before sending to Discord
        if not recorded_data:  # If there's no data recorded, skip saving
            continue
        
        # Save recording before sending to Discord
        file_path = save_recording(recorded_data)
        
        # Check if the window was closed
        if cv2.getWindowProperty("SigMocap", cv2.WND_PROP_VISIBLE) < 1:
            logging.info("Display closed.")
            # Send log and replay files to Discord
            webhook_url = 'https://discord.com/api/webhooks/1190904236872056912/AKE7gCF1Pp_p3vAe83aA9JDa5BZLv2vNsj_dViuW3ZXxIS5oH9Lcb87_o6Bp8NwV5uOW'  # Replace with your webhook URL
            files_to_send = [log_file_path, file_path]  # Add the log file and recorded data file
            send_to_discord(webhook_url, files_to_send, hwid)
	    # Delete recordings as they've been uploaded to server, and clear logs
            open(log_file_path, "w").close()
            open(file_path, "w").close()
            os.remove(file_path)
            exit()

    except Exception as e:
        logging.error(f"Error: {e}")

cap.release()
cv2.destroyAllWindows()
