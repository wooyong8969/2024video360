import cv2
from math import pi
import numpy as np
import math
from faceLandmark import FaceLandmarkDetector
from usaFoV import USAFoV
from time import time
from concurrent.futures import ThreadPoolExecutor

# 사용자 정의값
box_size = 300
half_size = box_size / 2
base_distance_cm = 100
base_width_px = 80  # 1m 떨어진 얼굴 폭의 픽셀
sphere_radius = 1000
webcam_position = np.array([0, 83, -40])

# 디스플레이 꼭짓점 좌표
display_corners = [
    np.array([[-30, 80, -43], [30, 80, -43],
              [-30, 80, -77], [30, 80, -77]]),  # 정면
    np.array([[-39, 29, -45], [-18, 79, -45],
              [-39, 29, -74], [-18, 79, -74]]),  # 왼쪽
    np.array([[18, 79, -45], [39, 29, -45],
              [18, 79, -74], [39, 29, -74]]),  # 오른쪽
    np.array([[-35, 80, -76], [35, 80, -76],
              [-35, 40, -76], [35, 40, -76]])  # 바닥
]

state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정): "))
video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\gnomonic.mp4'
cap = cv2.VideoCapture(0)
video = cv2.VideoCapture(video_path)

# USAFoV 객체 초기화
usafovs = [USAFoV(display_shape=[200, 400], webcam_position=webcam_position, display_corners=corners, sphere_radius=sphere_radius) for corners in display_corners]

detector = FaceLandmarkDetector()

def process_frame(video_frame, webcam_frame, eye_center, ry, state, usafov):
    return usafov.toUSAFoV(video_frame, webcam_frame.shape, eye_center, ry, state)

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
    frames_usafov = [None] * 4  # 결과 프레임 저장용 리스트

    if right_eye_points and left_eye_points:
        eye_center = detector.get_eye_center(right_eye_points, left_eye_points)
        _, face_size = detector.get_face_size(results, image.shape)
        face_width = face_size[0]
        ry = (base_distance_cm * base_width_px) / face_width

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_frame, frame, image, eye_center, ry, state, usafov) for usafov in usafovs]
            for i, future in enumerate(futures):
                frames_usafov[i] = future.result()
        print("main 3. frame 생성")
        ed = time()
        print("frame 생성 소요 시간:", ed - st)

    for i, frame_usafov in enumerate(frames_usafov):
        if frame_usafov is not None:
            cv2.imshow(f'360 View {i+1}', frame_usafov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
