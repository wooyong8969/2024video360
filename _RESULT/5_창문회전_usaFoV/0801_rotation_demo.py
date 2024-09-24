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

base_distance_cm = 45
base_width_px = 126# 1m 떨어진 얼굴 폭의 픽셀

monitor_width = 35
monitor_height = 23.5

display_distance = 50  # video frame 원점에서 display frame 원점까지의 거리
sphere_radius = 1000

webcam_position = np.array([0, 50, monitor_height / 2]) # 웹캠 좌표 (video frame)

# 디스플레이 꼭짓점 좌표 (video frame)
display_corners = np.array([
    [-monitor_width / 2, 50, monitor_height / 2],
    [monitor_width / 2, 50, monitor_height / 2],
    [-monitor_width / 2, 50, -monitor_height / 2],
    [monitor_width / 2, 50, -monitor_height / 2]
])


'''
# 제자리에서 회전
center_of_display = np.mean(display_corners, axis=0)
display_corners_centered = display_corners - center_of_display

theta = np.radians(-30)
rotation_matrix_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

rotated_corners  = np.dot(display_corners_centered, rotation_matrix_z.T)  # 시계방향으로 30도 회전한 디스플레이의 네 꼭짓점 좌표
display_corners = rotated_corners + center_of_display
'''

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정): "))

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\gnomonic.mp4'
cap = cv2.VideoCapture(0)
#640 480
video = cv2.VideoCapture(video_path) 

usafov = USAFoV(display_shape=[1080,1920], webcam_position=webcam_position, display_corners=display_corners, display_distance=display_distance, sphere_radius=sphere_radius)

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
    print("###################################################################################")
    print("main 1. 이미지 전처리 완료")
    
    right_eye_points, left_eye_points = detector.draw_landmarks(image, results)

    if right_eye_points and left_eye_points:
        eye_center = detector.get_eye_center(right_eye_points, left_eye_points)

        _, face_size = detector.get_face_size(results, image.shape)
        face_width = face_size[0]
        
        ry = (base_distance_cm * base_width_px) / face_width  # 모니터와 사람 사이의 거리 (cm단위)
        print("main 2. 모니터-사용자 거리 계산", ry)
        
        frame_usafov = usafov.toUSAFoV(frame, image.shape, eye_center, ry, state)
        print("main 3. frame 생성")
        ed = time()
        print("frame 생성 소요 시간:", ed - st)
    else:
        frame_usafov = usafov.toUSAFoV(frame, image.shape, [320, 240], ry, state)

    cv2.imshow('360 View', frame_usafov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()