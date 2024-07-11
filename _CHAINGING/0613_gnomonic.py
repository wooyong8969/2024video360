"""
MAIN으로 수정 중인 파일입니다!


"""


"""/**얼굴 landmark 탐지**/"""
import cv2
import mediapipe as mp 
import numpy as np 

class FaceLandmarkDetector:
    def __init__(self):
        # MediaPipe의 얼굴 mesh 관련 설정 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # 탐지할 수 있는 얼굴의 최대 개수
            refine_landmarks=True,  # 세부 landmark를 개선할 것인지 여부
            min_detection_confidence=0.5,  # 탐지 확신도의 최소값
            min_tracking_confidence=0.5)  # 추적 확신도의 최소값
        
        # landmark와 edges을 그리기 위한 설정 초기화
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # 오른쪽 눈과 왼쪽 눈의 landmark 인덱스 정의
        # 제 3자의 입장에서, 9시 방향부터 시계방향으로 / -3, -2, -1은 눈동자 좌표
        self.RIGHT_EYE_INDEX = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 476, 473, 474]
        self.LEFT_EYE_INDEX = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 471, 468, 469]

    '''이미지 처리'''
    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지를 BGR에서 RGB로 변환
        results = self.face_mesh.process(image_rgb)  # 얼굴 메시 처리 수행
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # 결과 이미지를 다시 BGR로 변환
        return results, image_bgr

    '''landmark 그리기'''
    def draw_landmarks(self, image, results):
        right_eye_points = []
        left_eye_points = []  # 각 landmark의 좌표를 저장할 리스트 생성

        if results.multi_face_landmarks:  # 탐지된 얼굴 landmark가 있으면
            for face_landmarks in results.multi_face_landmarks:
                # 오른쪽 눈 landmark 표시
                for idx in self.RIGHT_EYE_INDEX:
                    x, y = int(face_landmarks.landmark[idx].x * image.shape[1]), int(face_landmarks.landmark[idx].y * image.shape[0])
                    right_eye_points.append((x, y))
                    cv2.circle(image, (x, y), 3, (255, 0, 0), -1)  # 3크기의 속이 채워진 파란색 원 그리기
                # 왼쪽 눈 landmark 표시
                for idx in self.LEFT_EYE_INDEX:
                    x, y = int(face_landmarks.landmark[idx].x * image.shape[1]), int(face_landmarks.landmark[idx].y * image.shape[0])
                    left_eye_points.append((x, y))
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 3크기의 속이 채워진 초록색 원 그리기
                
                # 얼굴의 landmark와 edges 그리기
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,  # 이미 정의된 스타일로 landmark 그리기
                    connection_drawing_spec=self.drawing_spec) # 이미 정의된 스타일로 edges 그리기

        return right_eye_points, left_eye_points
    
    '''눈의 중앙 좌표 구하기'''
    def get_eye_center(self, right_eye_points, left_eye_points):
        # 각 눈 관련 좌표들의 평균값을 눈의 중앙 좌표라 가정
        right_eye_center = np.mean(right_eye_points, axis=0)
        left_eye_center = np.mean(left_eye_points, axis=0)
        
        eye_center = (right_eye_center + left_eye_center) / 2  # 두 눈의 좌표의 중앙 구하기
        return eye_center
    

    '''얼굴의 경계 좌표 구하기'''
    def get_face_bounds(self, results, image_shape):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [landmark.x * image_shape[1] for landmark in face_landmarks.landmark]
                ys = [landmark.y * image_shape[0] for landmark in face_landmarks.landmark]
                min_x, max_x = int(min(xs)), int(max(xs))
                min_y, max_y = int(min(ys)), int(max(ys))
                return min_x, min_y, max_x, max_y
        return None

    '''얼굴 크기 구하기'''
    def get_face_size(self, min_x, min_y, max_x, max_y):
        width = max_x - min_x
        height = max_y - min_y
        return width, height


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""/**LatLong Image에서 사용자가 정의한 중심점을 기준으로 NFOV 이미지 생성**/"""

'''
The MIT License (MIT)
Copyright (c) 2021 Vít Ambrož
https://github.com/VitaAmbroz/360Tracking

'''
from cv2 import cv2
from math import pi
import numpy as np

class NFOV():
    def __init__(self, height=400, width=800):
        self.FOV = [0.5, 0.5]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.height = height
        self.width = width

        self.screen_points = self._get_screen_img()
        self.convertedScreenCoord = None
        self.sphericalCoord = None
        self.sphericalCoordReshaped = None

    '''라디안 좌표로 변환'''
    def _get_coord_rad_point(self, point):
        return (point * 2 - 1) * np.array([self.PI, self.PI_2])

    def _get_coord_rad_screen_points(self):
        return (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (np.ones(self.screen_points.shape) * self.FOV)

    '''그리드 생성'''
    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T 
    
    '''구면 좌표를 그노모닉(직각) 좌표로 변환'''
    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    '''NFoV 추출'''
    def toNFOV(self, frame, center_point, computeRectPoints=False):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        self.cp = self._get_coord_rad_point(point=center_point)

        self.convertedScreenCoord = self._get_coord_rad_screen_points()
        self.sphericalCoord = self._calcSphericaltoGnomonic(self.convertedScreenCoord)

        self.sphericalCoordReshaped = self.sphericalCoord.reshape(self.height, self.width, 2).astype(np.float32) % 1

        out = cv2.remap(self.frame, (self.sphericalCoordReshaped[..., 0] * self.frame_width), (self.sphericalCoordReshaped[..., 1] * self.frame_height), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""/**'MAIN 실행문**/"""
'''
1. dx, dy, dx 계산 (모니터 상에서 픽셀 변화)
2. 1번 이용하여 진짜 이동거리 drx, dry, drz 계산 (현실 공간에서 위치 변화)
3. 2번 이용하여 창문 좌표 이동 (? - drx, ? - dry, ? - drz)

for(4~6): 모니터(창문) 모든 픽셀에 대해 반복, 선형 보간 이용
4. 3번에서 구한 좌표를 극좌표로 변환
5. 4번에서 구한 theta, phi 및 동영상 배경 거리 R 이용하여 목표 좌표 계산 (R, theta, phi)
6. 해당 목표 좌표 출력



'''

import math

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\0604_black_win.mp4'
cap = cv2.VideoCapture(0) 
video = cv2.VideoCapture(video_path) 

nfov = NFOV(height=400, width=800)
detector = FaceLandmarkDetector()


def calculate_dx(eye_center, frame_width):
    screen_center = frame_width / 2
    dx = (eye_center[0] - screen_center) / frame_width
    return dx

def calculate_dy(distance_R=100, base_distance_cm=30, base_width_px=200):
    bounds = detector.get_face_bounds(results, frame.shape)
    if bounds:
        min_x, min_y, max_x, max_y = bounds
        face_width, face_height = detector.get_face_size(min_x, min_y, max_x, max_y)
    
    dy = (base_distance_cm * base_width_px) / face_width
    return dy

def calculate_dz(eye_center, frame_height):
    screen_center = frame_height / 2
    dz = (eye_center[1] - screen_center) / frame_height
    return dz


pass

def 

def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    return r, theta, phi


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

    pass
    
    frame_nfov = cv2.flip(frame_nfov, 1)
    cv2.imshow('360 View', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()