import cv2
from math import pi
import numpy as np
import math
from faceLandmark import FaceLandmarkDetector
from usaFoV import USAFoV
from time import time
from concurrent.futures import ThreadPoolExecutor

'''사용자 정의값들'''

box_size = 300
half_size = box_size / 2

base_distance_cm = 100
base_width_px = 80  # 1m 떨어진 얼굴 폭의 픽셀

display_distance1 = 50  # video frame 원점에서 display frame 원점까지의 거리
sphere_radius = 1000

webcam_position = np.array([0, 83, -40])


# 디스플레이 꼭짓점 좌표 (display frame)
'''1 - 정면'''
display_corners1 = np.array([
    [-30, 80, -43], [30, 80, -43],
    [-30, 80, 77], [30, 80, 77]
])

'''2 - 왼쪽'''
display_corners2 = np.array([
    [-39, 279, -45], [-18, 79, -45],
    [-39, 29, -74], [-18, 29, -74]
])

'''3 - 오른쪽'''
display_corners3 = np.array([
    [18, 79, -45], [39, 29, -45],
    [18, 79, -74], [39, 79, -74]
])

'''4 - 바닥'''
display_corners4 = np.array([
    [-35, 80, -76], [35, 80, -76],
    [-35, 40, -76], [35, 40, -76]
])


state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정): "))

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\gnomonic.mp4'

cap = cv2.VideoCapture(0)
video = cv2.VideoCapture(video_path)

usafov1 = USAFoV(display_shape=[800, 1600], webcam_position=webcam_position, display_corners=display_corners1, display_distance=display_distance1, sphere_radius=sphere_radius)
usafov2 = USAFoV(display_shape=[800, 1600], webcam_position=webcam_position, display_corners=display_corners2, display_distance=display_distance2, sphere_radius=sphere_radius)
usafov3 = USAFoV(display_shape=[800, 1600], webcam_position=webcam_position, display_corners=display_corners3, display_distance=display_distance3, sphere_radius=sphere_radius)
usafov4 = USAFoV(display_shape=[800, 1600], webcam_position=webcam_position, display_corners=display_corners4, display_distance=display_distance4, sphere_radius=sphere_radius)

detector = FaceLandmarkDetector()

def process_frame(video_frame, webcam_frame, eye_center, ry, state):
    frame_usafov1 = usafov1.toUSAFoV(video_frame, webcam_frame.shape, eye_center, ry, state)
    frame_usafov2 = usafov2.toUSAFoV(video_frame, webcam_frame.shape, eye_center, ry, state)
    frame_usafov3 = usafov3.toUSAFoV(video_frame, webcam_frame.shape, eye_center, ry, state)
    frame_usafov4 = usafov4.toUSAFoV(video_frame, webcam_frame.shape, eye_center, ry, state)
    return frame_usafov1, frame_usafov2, frame_usafov3, frame_usafov4

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
    print("main 1. 이미지 전처리 완료")

    right_eye_points, left_eye_points = detector.draw_landmarks(image, results)

    if right_eye_points and left_eye_points:
        eye_center = detector.get_eye_center(right_eye_points, left_eye_points)

        _, face_size = detector.get_face_size(results, image.shape)
        face_width = face_size[0]

        ry = (base_distance_cm * base_width_px) / face_width  # 모니터와 사람 사이의 거리 (cm단위)
        print("main 2. 모니터-사용자 거리 계산", ry)

        with ThreadPoolExecutor() as executor:
            future = executor.submit(process_frame, frame, image, eye_center, ry, state)
            frame_usafov1, frame_usafov2, frame_usafov3, frame_usafov4 = future.result()
        print("main 3. frame 생성")
        ed = time()
        print("frame 생성 소요 시간:", ed - st)
    else:
        frame_usafov1 = usafov1.toUSAFoV(frame, image.shape, [320, 240], ry, state)
        frame_usafov2 = usafov2.toUSAFoV(frame, image.shape, [320, 240], ry, state)
        frame_usafov3 = usafov3.toUSAFoV(frame, image.shape, [320, 240], ry, state)
        frame_usafov4 = usafov4.toUSAFoV(frame, image.shape, [320, 240], ry, state)

    cv2.imshow('360 View 1', frame_usafov1)
    cv2.imshow('360 View 2', frame_usafov2)
    cv2.imshow('360 View 3', frame_usafov3)
    cv2.imshow('360 View 4', frame_usafov4)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
