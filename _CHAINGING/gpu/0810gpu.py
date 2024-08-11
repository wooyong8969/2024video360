import cv2
from math import pi
import cupy as cp
import math
from faceLandmark import FaceLandmarkDetector
from usaFoVg import USAFoV
from time import time

'''사용자 정의값들'''

box_size = 300
half_size = box_size / 2

base_distance_cm = 100
base_width_px = 80  # 1m 떨어진 얼굴 폭의 픽셀

monitor_width = 35
monitor_height = 23.5

display_distance = 50  # video frame 원점에서 display frame 원점까지의 거리
sphere_radius = 3000

webcam_position = cp.array([0, 50, monitor_height / 2])  # 웹캠 좌표 (video frame)

# 디스플레이 꼭짓점 좌표 (video frame)
display_corners = cp.array([
    [-monitor_width / 2, 50, monitor_height / 2],
    [monitor_width / 2, 50, monitor_height / 2],
    [-monitor_width / 2, 50, -monitor_height / 2],
    [monitor_width / 2, 50, -monitor_height / 2]
])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정, 3: 거울 모드, 4: 투명 모드): "))

video_path = r'C:\Users\user\Desktop\2024window\gnomonic.mp4'
cap0 = cv2.VideoCapture(1)  # 노트북 웹캠
cap2 = cv2.VideoCapture(0)  # 정면 웹캠
cap3 = cv2.VideoCapture(2)  # 후면 웹캠
video = cv2.VideoCapture(video_path)

usafov = USAFoV(display_shape=[800, 1600],
                webcam_position=webcam_position,
                display_corners=display_corners,
                sphere_radius=sphere_radius)

detector = FaceLandmarkDetector()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

while True:
    st = time()

    if state in [1, 2]:
        ret, frame = video.read()
        success, image = cap2.read()
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

        try:
            # Assuming `toUSAFoV` returns a CuPy array
            frame_usafov = usafov.toUSAFoV(frame, image.shape, eye_center, ry, state)

            # Ensure conversion to NumPy array
            if isinstance(frame_usafov, cp.ndarray):
                frame_usafov = frame_usafov.get()  # Convert to NumPy array
            ed = time()
            print("main 3. frame 생성 완료", ed - st)
        except Exception as e:
            print(f"Error during frame processing: {e}")
            break
    else:
        frame_usafov = usafov.toUSAFoV(frame, image.shape, [320, 240], ry, state)

        # Ensure conversion to NumPy array
        if isinstance(frame_usafov, cp.ndarray):
            frame_usafov = frame_usafov.get()

    cv2.imshow('360 View', frame_usafov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap0.release()
cap2.release()
cap3.release()
video.release()
cv2.destroyAllWindows()
