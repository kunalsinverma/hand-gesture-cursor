import cv2
import mediapipe as mp
import pyautogui

print("---System and Library Verification---")
print(f"OpenCV Version: {cv2.__version__}")
print(f"MediaPipe Version: {pyautogui.__version__}")

screen_width, screen_height = pyautogui.size()
print(f"Primary Screen Resolution : {screen_width}*{screen_height} pixels")
print("Environment setup is fully functiuonal!")