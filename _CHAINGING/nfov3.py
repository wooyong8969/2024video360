'''/**메인 내용 구현 시도 중**/''' 

import numpy as np
import cv2
from math import pi
from face_landmark2 import FaceLandmarkDetector

class SphereIntersectionCalculator:
    def __init__(self, R, box_height, box_width):
        self.R = R
        self.half_height = box_height / 2
        self.half_width = box_width / 2
        self.window_corners = np.array([
            [self.half_width, self.half_height, self.half_width],
            [-self.half_width, self.half_height, self.half_width],
            [self.half_width, self.half_height, -self.half_width],
            [-self.half_width, self.half_height, -self.half_width]
        ])
    
    def calculate_intersections(self, dx, dy):
        position = np.array([dx, dy, 0])
        intersection_points = []
        for corner in self.window_corners:
            direction = corner - position
            norm_direction = direction / np.linalg.norm(direction)
            
            a = np.dot(norm_direction, norm_direction)
            b = 2 * np.dot(norm_direction, position)
            c = np.dot(position, position) - self.R**2

            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                t2 = (-b + np.sqrt(discriminant)) / (2 * a)
                if t1 > 0 and t2 > 0:
                    t = min(t1, t2)
                elif t1 > 0:
                    t = t1
                elif t2 > 0:
                    t = t2
                else:
                    continue
                intersection_point = position + t * norm_direction
                intersection_points.append(intersection_point)
        return intersection_points

class NFOV:
    def __init__(self, height, width):
        self.FOV = [0.5, 0.5]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.height = height
        self.width = width

    def _get_coord_rad_point(self, point):
        return (point * 2 - 1) * np.array([self.PI, self.PI_2])

    def toNFOV(self, frame, intersections):
        frame_height, frame_width = frame.shape[:2]

        spherical_points = [self._get_coord_rad_point(intersection / self.R) for intersection in intersections]
        spherical_points = np.array(spherical_points)

        min_lon = spherical_points[:, 0].min()
        max_lon = spherical_points[:, 0].max()
        min_lat = spherical_points[:, 1].min()
        max_lat = spherical_points[:, 1].max()

        screen_x = (spherical_points[:, 0] - min_lon) / (max_lon - min_lon) * self.width
        screen_y = (spherical_points[:, 1] - min_lat) / (max_lat - min_lat) * self.height

        screen_x = screen_x.astype(int)
        screen_y = screen_y.astype(int)

        projected_img = np.zeros((self.height, self.width, 3), dtype=frame.dtype)
        for i in range(self.height):
            for j in range(self.width):
                x = min_lon + j / self.width * (max_lon - min_lon)
                y = min_lat + i / self.height * (max_lat - min_lat)
                if 0 <= x < frame_width and 0 <= y < frame_height:
                    projected_img[i, j] = frame[int(y), int(x)]

        return projected_img

# 초기 설정
height, width = 400, 800
R = 300
calculator = SphereIntersectionCalculator(R=R, box_height=height, box_width=width)
nfov = NFOV(height=height, width=width)
detector = FaceLandmarkDetector()

# 비디오 캡처 설정
video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\0528_test_video.mp4'  # 360도 비디오 파일 경로 설정
cap = cv2.VideoCapture(0)  # 웹캠 설정
video = cv2.VideoCapture(video_path)

def calculate_dx(eye_center, frame_width):
    screen_center = frame_width / 2
    dx = (eye_center[0] - screen_center) / frame_width * 300
    return dx

def calculate_dy():
    return 60  # dy의 값을 설정 (예제에서는 고정 값 사용)

while True:
    ret, frame = video.read()
    if not ret:
        print("영상 오류")
        break

    frame = cv2.flip(frame, 1)

    success, image = cap.read()
    if not success:
        print("웹캠 오류")
        break

    results, image = detector.process_frame(image)
    right_eye_points, left_eye_points = detector.draw_landmarks(image, results)
    if right_eye_points and left_eye_points:
        eye_center = detector.get_eye_center(right_eye_points, left_eye_points)
        dx = calculate_dx(eye_center, image.shape[1])
        dy = calculate_dy()
        intersections = calculator.calculate_intersections(dx, dy)
        frame_nfov = nfov.toNFOV(frame, intersections)
    else:
        frame_nfov = frame  # 랜드마크를 찾지 못한 경우 원본 프레임 사용

    cv2.imshow('360 View', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
