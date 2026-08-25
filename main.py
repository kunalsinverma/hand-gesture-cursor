import cv2
import mediapipe as mp
import pyautogui
import math

# Performance optimizations for real-time mouse control
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Get monitor's primary screen dimensions (e.g., 1920 x 1080)
screen_width, screen_height = pyautogui.size()

# Smoothing Configuration
SMOOTHING_FACTOR = 0.25
DEADZONE_PIXELS = 3.0

# State variables to store the previous cursor position across frames
prev_screen_x = 0.0
prev_screen_y = 0.0
is_first_detection = True

# Grace period: ignore brief 1-2 frame motion blurs during fast movement
HAND_LOST_THRESHOLD = 5
frames_hand_lost = 0

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Error: Failed to grab frame from camera.")
        break

    # 1. Flip horizontally for natural mirror view
    frame = cv2.flip(frame, 1)

    # 2. Get webcam frame dimensions (Height, Width, Channels)
    frame_height, frame_width, _ = frame.shape

    # 3. Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # If a hand is detected, draw the landmarks
    if results.multi_hand_landmarks:
        frames_hand_lost = 0  # Reset counter because hand is visible

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract the 4 key anchor landmarks forming the palm polygon:
            wrist = hand_landmarks.landmark[0]
            index_mcp = hand_landmarks.landmark[5]
            middle_mcp = hand_landmarks.landmark[9]
            pinky_mcp = hand_landmarks.landmark[17]

            # Calculate True Palm Centroid in normalized coordinates (0.0 to 1.0)
            palm_norm_x = (wrist.x + index_mcp.x + middle_mcp.x + pinky_mcp.x) / 4.0
            palm_norm_y = (wrist.y + index_mcp.y + middle_mcp.y + pinky_mcp.y) / 4.0

            # Convert to Webcam Pixel Coordinates (for drawing on the webcam feed)
            palm_cam_x = int(palm_norm_x * frame_width)
            palm_cam_y = int(palm_norm_y * frame_height)

            # Raw target coordinates on screen
            target_screen_x = palm_norm_x * screen_width
            target_screen_y = palm_norm_y * screen_height

            # Initialize previous position on first detection to avoid cursor jumping from (0,0)
            if is_first_detection:
                prev_screen_x = target_screen_x
                prev_screen_y = target_screen_y
                is_first_detection = False

            # Calculate distance moved since last frame (Euclidean Distance)
            distance_moved = math.hypot(target_screen_x - prev_screen_x, target_screen_y - prev_screen_y)

            # Only update cursor if movement exceeds the deadzone threshold
            if distance_moved > DEADZONE_PIXELS:
                # Apply Exponential Moving Average (EMA) formula
                smooth_screen_x = prev_screen_x + SMOOTHING_FACTOR * (target_screen_x - prev_screen_x)
                smooth_screen_y = prev_screen_y + SMOOTHING_FACTOR * (target_screen_y - prev_screen_y)

                # Move physical mouse cursor
                pyautogui.moveTo(int(smooth_screen_x), int(smooth_screen_y))

                # Update previous position for the next frame
                prev_screen_x = smooth_screen_x
                prev_screen_y = smooth_screen_y

            # Draw visual indicator on webcam feed
            cv2.circle(frame, (palm_cam_x, palm_cam_y), 12, (255, 255, 0), cv2.FILLED)

    else:
        # Increment lost counter when hand is missing
        frames_hand_lost += 1

        # Only reset when hand has genuinely left the camera (e.g. 5+ consecutive frames)
        if frames_hand_lost > HAND_LOST_THRESHOLD:
            is_first_detection = True


    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
camera.release()
cv2.destroyAllWindows()
hands.close()

