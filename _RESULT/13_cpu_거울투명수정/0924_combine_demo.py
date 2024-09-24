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
base_width_px = 126 # 1m 떨어진 얼굴 폭의 픽셀

sphere_radius = 1000

webcam_position = np.array([0, 45, -10]) # 웹캠 좌표 (video frame)

# 디스플레이 꼭짓점 좌표 (video frame)
display_corners = np.array([
    [-17, 45, -10], [17, 45, -10],
    [-17, 45, -35], [17, 45, -35]
])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정, 3: 거울 모드, 4: 투명 모드): "))

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\gnomonic.mp4'
capf = cv2.VideoCapture(0)  # 정면(front)
capb = cv2.VideoCapture(1)  # 후면(back)
video = cv2.VideoCapture(video_path)    # 360 영상

usafov = USAFoV(display_shape=[1080,1920],
                webcam_position=webcam_position,
                display_corners=display_corners,
                sphere_radius=sphere_radius)

detector = FaceLandmarkDetector()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

while True:
    st = time()
    
    if state in [1, 2]:
        ret, frame = video.read()
        success, image = capf.read()
    elif state == 3:
        ret, frame = capf.read()
        success, image = capf.read()
    elif state == 4:
        ret, frame = capb.read()
        frame = cv2.flip(frame, 1)
        success, image = capf.read()
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