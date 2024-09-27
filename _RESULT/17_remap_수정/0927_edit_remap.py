import cv2
import cupy as cp
import numpy as np
from math import pi
from faceLandmark import FaceLandmarkDetector
from usaFoV2 import USAFoV
from time import time
from screeninfo import get_monitors


'''사용자 정의값들'''

# ry 계산 위한 변수들
base_distance_cm = 70
base_width_px = 115
sphere_radius = 1000
z = 0

# 웹캠 관련 변수들
webcam_position = cp.array([0, 50, 8 + z])
webcam_D = 360
webcam_ratio = 480 / 640

horizon_tan = cp.float32(webcam_D) * cp.tan(cp.pi / cp.float32(8))
vertical_tan = cp.float32(webcam_D) * cp.tan(cp.pi / cp.float32(8)) * webcam_ratio
webcam_info = [webcam_position, horizon_tan, vertical_tan]

# 모니터 관련 정보
monitors = get_monitors()
monitor1 = monitors[0]
monitor_width1 = monitor1.width
monitor_height1 = monitor1.height

# 11cm
display_corners = [
    cp.array([[-5.5, 50, 7 + z], [5.5, 50, 7 + z],
             [-5.5, 50, 0 + z], [5.5, 50, 0 + z]])  # 1번 모니터
]

print(2)

#display_shapes = [(monitor1.height, monitor1.width)]
display_shapes = [(480, 800)]

if len(monitors) > 1:
    monitor2 = monitors[1]
    monitor_width2 = monitor2.width
    monitor_height2 = monitor2.height
    display_shapes.append((monitor2.height, monitor2.width))

    display_corners.append(
        cp.array([[-19.5, 19, 14 + z], [19.5, 19, 14 + z],
                  [-19.5, 19, 0 + z], [19.5, 19, 0 + z]])  # 2번 모니터
    )


'''초기 설정'''

preferred_eye = int(input("주시안을 입력해 주세요. (1: 오른쪽, 2: 왼쪽): "))
state = int(input("원하는 모드를 선택해 주세요. (1: 사용자 고정, 2: 디스플레이 고정, 3: 거울 모드, 4: 투명 모드): "))

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\gnomonic.mp4'
capf = cv2.VideoCapture(0)  # 정면 웹캠
capb = cv2.VideoCapture(2)  # 후면 웹캠
video = cv2.VideoCapture(video_path)
ret, frame = video.read()

# 각 화면에 대한 USAFoV 객체 생성
usafov_1 = USAFoV(display_shape=display_shapes[0],
                  webcam_info=webcam_info,
                  display_corners=display_corners[0],
                  sphere_radius=sphere_radius)

if len(monitors) > 1:
    usafov_2 = USAFoV(display_shape=display_shapes[1],
                      webcam_info=webcam_info,
                      display_corners=display_corners[1],
                      sphere_radius=sphere_radius)

detector = FaceLandmarkDetector()

'''모니터 조정 설정'''

rotation_angle = cp.pi / 180  # 회전 각도 1도

import cupy as cp
import numpy as np

def rotate_around_center(corners, axis, angle):
    try:
        corners_np = corners.get()

        center = np.mean(corners_np, axis=0)
        shifted = corners_np - center

        # 회전 행렬 생성
        if axis == 'x':
            rotation_matrix = np.array([[1, 0, 0],
                                        [0, np.cos(angle), -np.sin(angle)],
                                        [0, np.sin(angle), np.cos(angle)]], dtype=np.float32)
        elif axis == 'y':
            rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                        [0, 1, 0],
                                        [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
        elif axis == 'z':
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]], dtype=np.float32)

        rotated = np.dot(shifted, rotation_matrix.T)
        result_np = rotated + center

        result_cp = cp.asarray(result_np)

        return result_cp

    except Exception as e:
        print(f"Error in rotate_around_center: {e}")
        return corners  # 오류 발생 시 원래 배열 반환


def adjust_monitor(key, corners):
    move_step = cp.float32(0.5)
    monitor_changed = False

    if key == ord('8'):  # 상
        corners[:, 2] += move_step
        monitor_changed = True
    elif key == ord('2'):  # 하
        corners[:, 2] -= move_step
        monitor_changed = True
    elif key == ord('4'):  # 좌
        corners[:, 0] -= move_step
        monitor_changed = True
    elif key == ord('6'):  # 우
        corners[:, 0] += move_step
        monitor_changed = True
    elif key == ord('9'):  # 앞
        corners[:, 1] += move_step
        monitor_changed = True
    elif key == ord('1'):  # 뒤
        corners[:, 1] -= move_step
        monitor_changed = True

    elif key == ord('s'):  # x축 양의 회전
        corners = rotate_around_center(corners, 'x', rotation_angle)
        monitor_changed = True
    elif key == ord('x'):  # x축 음의 회전
        corners = rotate_around_center(corners, 'x', -rotation_angle)
        monitor_changed = True
    elif key == ord('a'):  # z축 양의 회전
        corners = rotate_around_center(corners, 'z', rotation_angle)
        monitor_changed = True
    elif key == ord('z'):  # z축 음의 회전
        corners = rotate_around_center(corners, 'z', -rotation_angle)
        monitor_changed = True
    elif key == ord('y'):  # y축 양의 회전
        corners = rotate_around_center(corners, 'y', rotation_angle)
        monitor_changed = True
    elif key == ord('h'):  # y축 음의 회전
        corners = rotate_around_center(corners, 'y', -rotation_angle)
        monitor_changed = True

    if monitor_changed:
        print(f"Updated corners after adjustment: {corners}")

    return corners, monitor_changed

# 모니터 수정 플래그
adjust_monitor_1 = False
adjust_monitor_2 = False
monitor_updated = False


'''main loop'''

while True:
    st = time()

    # 카메라 상태에 따른 프레임 처리
    if state in [1, 2]:
        success, image = capf.read()
    elif state == 3:  # 거울모드
        ret, frame = capf.read()
        success, image = capf.read()
    elif state == 4:  # 투명모드
        ret, frame = capb.read()
        frame = cv2.flip(frame, 1)
        success, image = capf.read()
    else:
        print("잘못된 상태 값입니다.")
        break

    if not ret or not success:
        print("영상 오류")
        break

    # 얼굴 랜드마크 처리
    results, image = detector.process_frame(image)
    right_eye_points, left_eye_points = detector.draw_landmarks(image, results)

    # 사용자 거리 계산
    if right_eye_points and left_eye_points:
        eye_center = np.mean(right_eye_points, axis=0) if preferred_eye == 1 else np.mean(left_eye_points, axis=0)
        _, face_size = detector.get_face_size(results, image.shape)
        face_width = face_size[0]
        ry = (base_distance_cm * base_width_px) / face_width
    else:
        eye_center = [320, 240]
        ry = 70

    # 모니터 좌표 조정 시 객체 재생성
    if monitor_updated:
        usafov_1 = USAFoV(display_shape=display_shapes[0],
                          webcam_info=webcam_info,
                          display_corners=display_corners[0],
                          sphere_radius=sphere_radius)

        usafov_2 = USAFoV(display_shape=display_shapes[1],
                          webcam_info=webcam_info,
                          display_corners=display_corners[1],
                          sphere_radius=sphere_radius)
        print("USAFoV 객체 재생성 완료")
        monitor_updated = False

    # frame 생성
    try:
        frame_usafov_1 = usafov_1.toUSAFoV(frame, image.shape, eye_center, ry, state)
        if isinstance(frame_usafov_1, cp.ndarray):
            frame_usafov_1 = frame_usafov_1.get()

        if state in [1, 2]:
            frame_usafov_2 = usafov_2.toUSAFoV(frame, image.shape, eye_center, ry, state)
            if isinstance(frame_usafov_2, cp.ndarray):
                frame_usafov_2 = frame_usafov_2.get()
            cv2.imshow('2 View', frame_usafov_2)

        ed = time()
        print("frame 생성 시간", ed - st)
    except Exception as e:
        print(f"Error during frame processing: {e}")
        break

    cv2.imshow('1 View', frame_usafov_1)

    # 키 입력 처리
    key = cv2.waitKey(1)

    if key == ord('o'):  # 1번 모니터 조정 시작
        adjust_monitor_1 = True
        adjust_monitor_2 = False
        print("1번 모니터 조정 모드")

    elif key == ord('t'):  # 2번 모니터 조정 시작
        adjust_monitor_1 = False
        adjust_monitor_2 = True
        print("2번 모니터 조정 모드")

    elif key == ord('q'):  # 종료
        print("최종 모니터 좌표 출력:")
        print("1번 모니터 좌표: ", display_corners[0])
        print("2번 모니터 좌표: ", display_corners[1])
        break

    # 1번 모니터 좌표 조정
    if adjust_monitor_1:
        display_corners[0], monitor_changed = adjust_monitor(key, display_corners[0])
        if monitor_changed:
            monitor_updated = True

    # 2번 모니터 좌표 조정
    if adjust_monitor_2:
        display_corners[1], monitor_changed = adjust_monitor(key, display_corners[1])
        if monitor_changed:
            monitor_updated = True

capf.release()
capb.release()
video.release()
cv2.destroyAllWindows()
