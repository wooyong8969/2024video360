import cv2
import cupy as cp
from faceLandmark import FaceLandmarkDetector
from usaFoV import USAFoV
from time import time

'''사용자 정의값들'''

box_size = 300
half_size = box_size / 2
base_distance_cm = 80
base_width_px = 72
sphere_radius = 1000

webcam_position = cp.array([0, 83, -40])  # cupy 배열로 변경

# 디스플레이 꼭짓점 좌표
display_corners = [
    cp.array([[-30, 80, -43], [30, 80, -43],
              [-30, 80, -77], [30, 80, -77]]),  # 앞
    cp.array([[-39, 29, -45], [-18, 79, -45],
              [-39, 29, -74], [-18, 79, -74]]), # 왼쪽
    cp.array([[18, 79, -45], [39, 29, -45],
              [18, 79, -74], [39, 29, -74]]),   # 오른쪽
    cp.array([[-35, 80, -76], [35, 80, -76],
              [-35, 40, -76], [35, 40, -76]])   # 바닥
]

state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정): "))

video_path = r'C:\Users\user\Desktop\2024window\gnomonic.mp4'
cap = cv2.VideoCapture(0)
video = cv2.VideoCapture(video_path)

# 각 화면의 크기 정의
display_shapes = [(1080, 1920), (1080, 1920), (1080, 1920), (1440, 2560)]

# 각 화면에 대한 USAFoV 객체 생성
usafov_front = USAFoV(display_shape=display_shapes[0],
                      webcam_position=webcam_position,
                      display_corners=display_corners[0],
                      sphere_radius=sphere_radius)

usafov_left = USAFoV(display_shape=display_shapes[1],
                     webcam_position=webcam_position,
                     display_corners=display_corners[1],
                     sphere_radius=sphere_radius)

usafov_right = USAFoV(display_shape=display_shapes[2],
                      webcam_position=webcam_position,
                      display_corners=display_corners[2],
                      sphere_radius=sphere_radius)

usafov_bottom = USAFoV(display_shape=display_shapes[3],
                       webcam_position=webcam_position,
                       display_corners=display_corners[3],
                       sphere_radius=sphere_radius)

detector = FaceLandmarkDetector()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

while True:
    st = time()

    ret, frame = video.read()
    success, image = cap.read()

    if not ret or not success:
        print("영상 또는 웹캠 오류")
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
            # 앞, 왼쪽, 오른쪽, 바닥 화면 각각에 대해 프레임 생성
            frame_usafov_front = usafov_front.toUSAFoV(frame, image.shape, eye_center, ry, state)
            frame_usafov_left = usafov_left.toUSAFoV(frame, image.shape, eye_center, ry, state)
            frame_usafov_right = usafov_right.toUSAFoV(frame, image.shape, eye_center, ry, state)
            frame_usafov_bottom = usafov_bottom.toUSAFoV(frame, image.shape, eye_center, ry, state)

            # CuPy 배열인 경우 NumPy 배열로 변환
            if isinstance(frame_usafov_front, cp.ndarray):
                frame_usafov_front = frame_usafov_front.get()
            if isinstance(frame_usafov_left, cp.ndarray):
                frame_usafov_left = frame_usafov_left.get()
            if isinstance(frame_usafov_right, cp.ndarray):
                frame_usafov_right = frame_usafov_right.get()
            if isinstance(frame_usafov_bottom, cp.ndarray):
                frame_usafov_bottom = frame_usafov_bottom.get()
            ed = time()
            print("main 3. frame 생성 완료", ed - st)
        except Exception as e:
            print(f"Error during frame processing: {e}")
            break
    else:
        frame_usafov_front = usafov_front.toUSAFoV(frame, image.shape, [320, 240], ry, state)
        frame_usafov_left = usafov_left.toUSAFoV(frame, image.shape, [320, 240], ry, state)
        frame_usafov_right = usafov_right.toUSAFoV(frame, image.shape, [320, 240], ry, state)
        frame_usafov_bottom = usafov_bottom.toUSAFoV(frame, image.shape, [320, 240], ry, state)

        # CuPy 배열인 경우 NumPy 배열로 변환
        if isinstance(frame_usafov_front, cp.ndarray):
            frame_usafov_front = frame_usafov_front.get()
        if isinstance(frame_usafov_left, cp.ndarray):
            frame_usafov_left = frame_usafov_left.get()
        if isinstance(frame_usafov_right, cp.ndarray):
            frame_usafov_right = frame_usafov_right.get()
        if isinstance(frame_usafov_bottom, cp.ndarray):
            frame_usafov_bottom = frame_usafov_bottom.get()

    # 각 프레임을 다른 창에 표시
    cv2.imshow('Front View', frame_usafov_front)
    cv2.imshow('Left View', frame_usafov_left)
    cv2.imshow('Right View', frame_usafov_right)
    cv2.imshow('Bottom View', frame_usafov_bottom)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
