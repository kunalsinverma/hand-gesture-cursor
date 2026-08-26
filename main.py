import cv2
import math
import mediapipe as mp
import pyautogui

# 1. PyAutoGUI Performance Optimization
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# 2. Get Monitor Dimensions
screen_width, screen_height = pyautogui.size()

# 3. Motion & Interaction Box Configuration
SMOOTHING_FACTOR = 0.25
DEADZONE_PIXELS = 3.0
MARGIN_X = 0.20   # 20% horizontal margin for easy corner reach
MARGIN_Y = 0.20   # 20% vertical margin for easy taskbar reach

# 4. 3D Pinch-Click Ratio Configuration
PINCH_TRIGGER_RATIO = 0.22   # Pinch ratio threshold to trigger click
PINCH_RELEASE_RATIO = 0.30   # Ratio required to release click
is_pinched = False

# State variables for cursor motion
prev_screen_x = 0.0
prev_screen_y = 0.0
is_first_detection = True

HAND_LOST_THRESHOLD = 5
frames_hand_lost = 0

# 5. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

def get_3d_distance(p1, p2):
    """Calculates 3D Euclidean distance between two MediaPipe landmarks."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

while True:
    success, frame = camera.read()

    if not success:
        print("Error: Failed to grab frame from camera.")
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Draw the Active Interaction Box on the camera preview (Yellow Box)
    box_x1 = int(MARGIN_X * frame_width)
    box_y1 = int(MARGIN_Y * frame_height)
    box_x2 = int((1.0 - MARGIN_X) * frame_width)
    box_y2 = int((1.0 - MARGIN_Y) * frame_height)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        frames_hand_lost = 0

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # --- 1. Cursor Movement via Palm Centroid with Active Box Mapping ---
            wrist = hand_landmarks.landmark[0]
            index_mcp = hand_landmarks.landmark[5]
            middle_mcp = hand_landmarks.landmark[9]
            pinky_mcp = hand_landmarks.landmark[17]

            palm_norm_x = (wrist.x + index_mcp.x + middle_mcp.x + pinky_mcp.x) / 4.0
            palm_norm_y = (wrist.y + index_mcp.y + middle_mcp.y + pinky_mcp.y) / 4.0

            # Remap palm position from active margin box to full screen [0.0, 1.0]
            remapped_x = (palm_norm_x - MARGIN_X) / (1.0 - 2 * MARGIN_X)
            remapped_y = (palm_norm_y - MARGIN_Y) / (1.0 - 2 * MARGIN_Y)

            # Clamp coordinates so moving outside the box pins cursor to screen edge
            clamped_norm_x = max(0.0, min(1.0, remapped_x))
            clamped_norm_y = max(0.0, min(1.0, remapped_y))

            target_screen_x = clamped_norm_x * screen_width
            target_screen_y = clamped_norm_y * screen_height

            if is_first_detection:
                prev_screen_x = target_screen_x
                prev_screen_y = target_screen_y
                is_first_detection = False

            distance_moved = math.hypot(target_screen_x - prev_screen_x, target_screen_y - prev_screen_y)

            if distance_moved > DEADZONE_PIXELS:
                smooth_screen_x = prev_screen_x + SMOOTHING_FACTOR * (target_screen_x - prev_screen_x)
                smooth_screen_y = prev_screen_y + SMOOTHING_FACTOR * (target_screen_y - prev_screen_y)

                pyautogui.moveTo(int(smooth_screen_x), int(smooth_screen_y))

                prev_screen_x = smooth_screen_x
                prev_screen_y = smooth_screen_y

            # Draw Palm Centroid indicator
            palm_cam_x = int(palm_norm_x * frame_width)
            palm_cam_y = int(palm_norm_y * frame_height)
            cv2.circle(frame, (palm_cam_x, palm_cam_y), 10, (255, 255, 0), cv2.FILLED)

            # --- 2. 3D Pinch Click with Scale Normalization ---
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # 3D Distance between thumb & index
            pinch_dist_3d = get_3d_distance(thumb_tip, index_tip)

            # Hand Reference Size (Wrist to Middle Knuckle)
            hand_size_3d = get_3d_distance(wrist, middle_mcp)

            # Prevent division by zero
            pinch_ratio = pinch_dist_3d / (hand_size_3d + 1e-6)

            # Convert fingertip coordinates to webcam pixels for drawing
            thumb_px = (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height))
            index_px = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
            mid_x = int((thumb_px[0] + index_px[0]) / 2)
            mid_y = int((thumb_px[1] + index_px[1]) / 2)

            # State Machine using 3D Pinch Ratio
            if pinch_ratio < PINCH_TRIGGER_RATIO and not is_pinched:
                pyautogui.click()
                is_pinched = True
                print(f">>> CLICK! (Ratio: {pinch_ratio:.2f}) <<<")

            elif pinch_ratio > PINCH_RELEASE_RATIO and is_pinched:
                is_pinched = False

            # Visual Feedback
            line_color = (0, 255, 0) if is_pinched else (0, 0, 255)
            cv2.line(frame, thumb_px, index_px, line_color, 3)
            cv2.circle(frame, (mid_x, mid_y), 8, line_color, cv2.FILLED)

    else:
        frames_hand_lost += 1
        if frames_hand_lost > HAND_LOST_THRESHOLD:
            is_first_detection = True
            is_pinched = False

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
camera.release()
cv2.destroyAllWindows()
hands.close()


