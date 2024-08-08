import cv2
from math import pi
import numpy as np
import math
from faceLandmark import FaceLandmarkDetector
from gopusaFoV import USAFoV
from screeninfo import get_monitors
from time import time

'''사용자 정의값들'''

box_size = 300
half_size = box_size / 2

base_distance_cm = 100
base_width_px = 80 # 1m 떨어진 얼굴 폭의 픽셀

monitor_width = 35
monitor_height = 23.5

display_distance = 50  # video frame 원점에서 display frame 원점까지의 거리
sphere_radius = 3000

webcam_position = np.array([0, 50, monitor_height / 2]) # 웹캠 좌표 (video frame)

# 디스플레이 꼭짓점 좌표 (video frame)
display_corners = np.array([
    [-monitor_width / 2, 50, monitor_height / 2],
    [monitor_width / 2, 50, monitor_height / 2],
    [-monitor_width / 2, 50, -monitor_height / 2],
    [monitor_width / 2, 50, -monitor_height / 2]
])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정, 3: 거울 모드, 4: 투명 모드): "))

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\20240604능선.mp4'
cap0 = cv2.VideoCapture(0)  # 노트북 웹캠
cap2 = cv2.VideoCapture(2)  # 상단 USB 웹캠
cap3 = cv2.VideoCapture(3)  # 하단 USB 웹캠
video = cv2.VideoCapture(video_path) 

usafov = USAFoV(display_shape=[800,1600],
                webcam_position=webcam_position,
                display_corners=display_corners,
                display_distance=display_distance,
                sphere_radius=sphere_radius)

detector = FaceLandmarkDetector()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

while True:
    st = time()
    
    if state in [1, 2]:
        ret, frame = video.read()
        success, image = cap0.read()
    elif state == 3:
        ret, frame = cap2.read()
        success, image = cap2.read()
    elif state == 4:
        ret, frame = cap3.read()
        frame = cv2.flip(frame, 1)
        success, image = cap2.read()
    else:
        print("잘못된 상태 값입니다.")
        break

    if not ret:
        print("영상 오류")
        break

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

cap0.release()
cap2.release()
cap3.release()
video.release()
cv2.destroyAllWindows()
