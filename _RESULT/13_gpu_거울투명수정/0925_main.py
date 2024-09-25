import cv2
from math import pi
import cupy as cp
import math
from faceLandmark import FaceLandmarkDetector
from usaFoV2 import USAFoV
from time import time

'''사용자 정의값들'''

# ry 계산 위한 변수들
base_distance_cm = 70
base_width_px = 130
sphere_radius = 1000

# 웹캠 관련 변수들
webcam_position = cp.array([0, 90, 0])
webcam_D = 360
webcam_ratio = 9 / 16

horizon_tan = cp.float32(webcam_D) * cp.tan(cp.pi / cp.float32(4))
vertical_tan = cp.float32(webcam_D) * cp.tan(cp.pi / cp.float32(4)) * webcam_ratio
webcam_info = [webcam_position, horizon_tan, vertical_tan]

display_corners = cp.array([[-31, 90, -3], [31, 90, -3],
             [-31, 90, -37], [31, 90, -37]])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정, 3: 거울 모드, 4: 투명 모드): "))

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\gnomonic.mp4'
capf = cv2.VideoCapture(0)  # 정면 웹캠
capb = cv2.VideoCapture(1)  # 후면 웹캠
video = cv2.VideoCapture(video_path)

usafov = USAFoV(display_shape=[800, 1600],
                webcam_info=webcam_info,
                display_corners=display_corners,
                sphere_radius=sphere_radius)

detector = FaceLandmarkDetector()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

while True:
    st = time()

    if state in [1, 2]:
        ret, frame = video.read()
        success, image = capf.read()
    elif state == 3:    # 거울모드
        ret, frame = capf.read()
        frame = cv2.flip(frame, 0)
        success, image = capf.read()
    elif state == 4:    # 투명모드
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

        try:
            frame_usafov = usafov.toUSAFoV(frame, image.shape, eye_center, ry, state)

            if isinstance(frame_usafov, cp.ndarray):
                frame_usafov = frame_usafov.get()
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

capf.release()
capb.release()
video.release()
cv2.destroyAllWindows()
