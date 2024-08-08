"""/**MAIN 파일**/"""
import cv2
from math import pi
import numpy as np
import math
from faceLandmark import FaceLandmarkDetector
from usaFoV import USAFoV
from time import time

'''사용자 정의값들'''

box_size = 300
half_size = box_size / 2

base_distance_cm = 100
base_width_px = 80 # 1m 떨어진 얼굴 폭의 픽셀

monitor_width = 50
monitor_height = 25

sphere_radius = 1000

webcam_position = np.array([0, 50, monitor_height])

print(1)

# 디스플레이 꼭짓점 좌표 (display frame)
display_corners1 = np.array([
    [-monitor_width / 2, 50, monitor_height / 2],
    [0, 50, monitor_height / 2],
    [-monitor_width / 2, 50, -monitor_height / 2],
    [0, 50, -monitor_height / 2]
])

display_corners2 = np.array([
    [0, 50, monitor_height / 2],
    [monitor_width / 2, 50, monitor_height / 2],
    [0, 50, -monitor_height / 2],
    [monitor_width / 2, 50, -monitor_height / 2]
])

print(2)

theta = np.radians(90)
rotation_matrix_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

display_corners1 =  np.dot(display_corners1, rotation_matrix_z.T)
display_corners2 =  np.dot(display_corners2, rotation_matrix_z.T)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정): "))

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\gnomonic.mp4'
#video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\20240604능선.mp4'

cap = cv2.VideoCapture(0)
#640 480
video = cv2.VideoCapture(video_path) 

usafov1 = USAFoV(display_shape=[400, 800], webcam_position=webcam_position, display_corners=display_corners1, sphere_radius=sphere_radius)
usafov2 = USAFoV(display_shape=[400, 800], webcam_position=webcam_position, display_corners=display_corners2, sphere_radius=sphere_radius)

detector = FaceLandmarkDetector()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

while True:
    st = time()
    ret, frame = video.read()
    if not ret:
        print("영상 오류")
        break

    success, image = cap.read()
    if not success:
        print("웹캠 오류")
        break

    results, image = detector.process_frame(image)
    print("webcam:", image.shape[0], image.shape[2])
    print("###################################################################################")
    print("main 1. 이미지 전처리 완료")
    
    right_eye_points, left_eye_points = detector.draw_landmarks(image, results)

    if right_eye_points and left_eye_points:
        eye_center = detector.get_eye_center(right_eye_points, left_eye_points)

        _, face_size = detector.get_face_size(results, image.shape)
        face_width = face_size[0]
        
        ry = (base_distance_cm * base_width_px) / face_width  # 모니터와 사람 사이의 거리 (cm단위)
        print("main 2. 모니터-사용자 거리 계산", ry)
        
        frame_usafov1 = usafov1.toUSAFoV(frame, image.shape, eye_center, ry, state)
        frame_usafov2 = usafov2.toUSAFoV(frame, image.shape, eye_center, ry, state)
        print("main 3. frame 생성")
        ed = time()
        print("frame 생성 소요 시간:", ed - st)
    else:
        frame_usafov1 = usafov1.toUSAFoV(frame, image.shape, [320, 240], ry, state)
        frame_usafov2 = usafov1.toUSAFoV(frame, image.shape, [320, 240], ry, state)

    cv2.imshow('360 View 1', frame_usafov1)
    cv2.imshow('360 View 2', frame_usafov2)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()