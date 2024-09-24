import cv2
import cupy as cp
from faceLandmark import FaceLandmarkDetector
from usaFoV import USAFoV
from time import time

'''사용자 정의값들'''

# ry 계산 위한 변수들
base_distance_cm = 73
base_width_px = 72
sphere_radius = 1000

# 웹캠 관련 변수들
webcam_position = cp.array([0, 90, 0])
horizon_tan = cp.float32(360) * cp.tan(cp.pi / cp.float32(4))
vertical_tan = cp.float32(360) * cp.tan(cp.pi / cp.float32(4)) * (9 / 16)
webcam_info = [webcam_position, horizon_tan, vertical_tan]

display_corners = [         # 2개
    cp.array([[-31, 90, -3], [31, 90, -3],
             [-31, 90, -37], [31, 90, -37]]),  # 1번
    cp.array([[-35.5, 89, -37.5], [35.5, 89, -37.5],
              [-35.5, 90-40, -37.5], [35.5, 90-40, -37.5]]), # 2번
]

state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정, 3: 거울 모드, 4: 투명 모드): "))

video_path = r'C:\Users\user\Desktop\2024window\20240828.jpg'
capf = cv2.VideoCapture(0)  # 정면 웹캠
capb = cv2.VideoCapture(1)  # 후면 웹캠
video = cv2.VideoCapture(video_path)

# 각 화면의 크기 정의
display_shapes = [(1920, 1080), (1920, 1080)]

# 각 화면에 대한 USAFoV 객체 생성
usafov_1 = USAFoV(display_shape=display_shapes[0],
                      webcam_position=webcam_position,
                      display_corners=display_corners[0],
                      sphere_radius=sphere_radius)

usafov_2 = USAFoV(display_shape=display_shapes[1],
                     webcam_position=webcam_position,
                     display_corners=display_corners[1],
                     sphere_radius=sphere_radius)

detector = FaceLandmarkDetector()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

while True:
    st = time()

    if state == 1 or state == 2:
        success, image = capf.read()
    elif state == 3:    # 거울
        ret, frame = capf.read()
        success, image = capf.read()
    elif state == 4:    # 투명
        ret, frame = capb.read()  # 후면 웹캠 사용
        frame = cv2.flip(frame, 1)  # 거울 모드처럼 뒤집기
        success, image = capf.read()  # 정면 웹캠 사용
    else:
        print("잘못된 상태 값입니다.")
        break

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
            frame_usafov_1 = usafov_1.toUSAFoV(frame, image.shape, eye_center, ry, state)
            frame_usafov_2 = usafov_2.toUSAFoV(frame, image.shape, eye_center, ry, state)

            # CuPy 배열인 경우 NumPy 배열로 변환
            if isinstance(frame_usafov_1, cp.ndarray):
                frame_usafov_1 = frame_usafov_1.get()
            if isinstance(frame_usafov_2, cp.ndarray):
                frame_usafov_2 = frame_usafov_2.get()

            ed = time()
            print("main 3. frame 생성 완료", ed - st)
        except Exception as e:
            print(f"Error during frame processing: {e}")
            break
    else:
        frame_usafov_1 = usafov_1.toUSAFoV(frame, image.shape, [320, 240], ry, state)
        frame_usafov_2 = usafov_2.toUSAFoV(frame, image.shape, [320, 240], ry, state)

        # CuPy 배열인 경우 NumPy 배열로 변환
        if isinstance(frame_usafov_1, cp.ndarray):
            frame_usafov_1 = frame_usafov_1.get()
        if isinstance(frame_usafov_2, cp.ndarray):
            frame_usafov_2 = frame_usafov_2.get()

    # 각 프레임을 다른 창에 표시
    cv2.imshow('1 View', frame_usafov_1)
    cv2.imshow('2 View', frame_usafov_2)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
