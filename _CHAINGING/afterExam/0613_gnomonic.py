"""
MAIN으로 수정 중인 파일입니다!

"""


from cv2 import cv2
from math import pi
import numpy as np
import math
from faceLandmark import FaceLandmarkDetector
from usaFoV import USAFoV

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

box_size = 300
half_size = box_size / 2

distance_R = 100
base_distance_cm = 30
base_width_px = 200

# 웹캠 좌표 (display frame)
webcam_position = np.array([0, half_size, 0])

# 디스플레이 꼭짓점 좌표 (display frame)
display_corners = np.array([
    [half_size, half_size, half_size],
    [-half_size, half_size, half_size],
    [half_size, half_size, -half_size],
    [-half_size, half_size, -half_size]
])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\0604_black_win.mp4'
cap = cv2.VideoCapture(0) 
video = cv2.VideoCapture(video_path) 

nfov = USAFoV(height=800, width=1600, webcam_position=webcam_position, display_corners=display_corners)
detector = FaceLandmarkDetector()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


while True:
    ret, frame = video.read()
    if not ret:
        break

    success, image = cap.read()
    if not success:
        print("웹캠 오류")
        break

    results, image = detector.process_frame(image)

    right_eye_points, left_eye_points = detector.draw_landmarks(image, results)
    if right_eye_points and left_eye_points:
        eye_center = detector.get_eye_center(right_eye_points, left_eye_points)
        frame_nfov = nfov.toNFOV(frame, eye_center)
    else:
        frame_nfov = nfov.toNFOV(frame, eye_center)

    cv2.imshow('360 View', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()