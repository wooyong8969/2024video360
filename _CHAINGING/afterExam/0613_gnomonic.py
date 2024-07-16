"""/**MAIN 파일**/"""
from cv2 import cv2
from math import pi
import numpy as np
import math
from faceLandmark import FaceLandmarkDetector
from usaFoV import USAFoV
from screeninfo import get_monitors

'''사용자 정의값들'''

box_size = 300
half_size = box_size / 2

distance_R = 100
base_distance_cm = 30
base_width_px = 200

# 웹캠 좌표 (display frame)
webcam_position = np.array([0, half_size, 0])

monitor = get_monitors()[0]
monitor_width = monitor.width
monitor_height = monitor.height

# 디스플레이 꼭짓점 좌표 (display frame)
display_corners = np.array([
    [-monitor_width / 2, 0, monitor_height / 2],
    [monitor_width / 2, 0, monitor_height / 2],
    [-monitor_width / 2, 0, -monitor_height / 2],
    [monitor_width / 2, 0, -monitor_height / 2]
])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\0604_black_win.mp4'
cap = cv2.VideoCapture(0)
#640 470
video = cv2.VideoCapture(video_path) 

usafov = USAFoV(display_shape=[400,800], webcam_position=webcam_position, display_corners=display_corners)
detector = FaceLandmarkDetector()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


while True:
    ret, frame = video.read()
    if not ret:
        print("영상 오류")
        break

    success, image = cap.read()
    if not success:
        print("웹캠 오류")
        break

    results, image = detector.process_frame(image)
    print("1. 이미지 전처리 완료")

    right_eye_points, left_eye_points = detector.draw_landmarks(image, results)
    if right_eye_points and left_eye_points:
        eye_center = detector.get_eye_center(right_eye_points, left_eye_points)

        _, face_size = detector.get_face_size(results, image.shape)
        face_width = face_size[0]
        px_to_cm = base_distance_cm / base_width_px
        distance_cm = face_width * px_to_cm
        ry = -distance_cm  # 모니터와 사람 사이의 거리 (cm단위)
        print("2. 모니터-사용자 거리 계산", ry)
        
        frame_usafov = usafov.toUSAFoV(frame, image.shape, eye_center, ry)
        print("3. frame 생성")
    else:
        frame_usafov = usafov.toUSAFoV(frame, image.shape, eye_center, ry)

    cv2.imshow('360 View', frame_usafov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()