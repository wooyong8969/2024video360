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
    

    '''얼굴의 경계 좌표 및 크기 구하기'''
    def get_face_size(self, results, image_shape):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [landmark.x * image_shape[1] for landmark in face_landmarks.landmark]
                ys = [landmark.y * image_shape[0] for landmark in face_landmarks.landmark]
                min_x, max_x = int(min(xs)), int(max(xs))
                min_y, max_y = int(min(ys)), int(max(ys))
                width = max_x - min_x
                height = max_y - min_y
                return (min_x, min_y, max_x, max_y), (width, height)
        return None, None
    












"""/**LatLong Image에서 사용자가 정의한 중심점을 기준으로 USAFoV 이미지 생성**/"""

from cv2 import cv2
from math import pi
import numpy as np

class USAFoV():
    def __init__(self, height, width, webcam_position, display_corners):
        self.PI = pi
        self.PI_2 = pi * 0.5

        self.frame = None
        self.display = None
        self.display_height = height
        self.display_width = width

        self.webcam_position = webcam_position
        self.display_corners = display_corners

        self.plane_coeffs = self._calculate_plane_equation(display_corners)

    '''디스플레이 중앙 원점 - 사용자의 위치 계산'''
    def _calculate_position(self, eye_center, rz, webcam_theta, webcam_alpha):
        pass
        #return D_user_position

    '''사용자 원점 - 디스플레이의 네 모서리 좌표 계산'''
    def _calculate_corners(self, user_position):
        D_display_corners = self.display_corners
        U_display_corners = D_display_corners - user_position
        return U_display_corners

    '''사용자 원점 - 디스플레이 평면의 방정식 계산'''
    def _calculate_plane_equation(self, display_corners):
        p1, p2, p3 = display_corners[0], display_corners[1], display_corners[2]
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        n = np.cross(v1, v2)

        # Ax + By + Cz + D = 0
        A, B, C = n
        D = -np.dot(n, p1)

        return A, B, C, D

    '''사용자 원점 - 디스플레이 그리드 생성'''
    def _create_display_grid(self, U_display_corners):
        # USER_display_corners 순서: [Top-Right, Top-Left, Bottom-Right, Bottom-Left]
        A, B, C, D = self.plane_coeffs

        top_left = U_display_corners[1]
        bottom_right = U_display_corners[2]

        x1, y1, _ = top_left
        x2, y2, _ = bottom_right

        x_values = np.linspace(x1, x2, self.display_width)
        y_values = np.linspace(y1, y2, self.display_height)

        x_grid, y_grid = np.meshgrid(x_values, y_values, indexing='ij')

        z_grid = -(A * x_grid + B * y_grid + D) / C

        U_display_grid_shape = (self.display_width, self.display_height, 3)
        U_display_grid = np.empty(U_display_grid_shape, dtype=float)
        U_display_grid[..., 0] = x_grid
        U_display_grid[..., 1] = y_grid
        U_display_grid[..., 2] = z_grid

        return U_display_grid

    '''사용자 원점 - 디스플레이 그리드의 직각좌표를 구면 좌표로 변환'''
    def _convert_to_spherical(self, U_display_corners):
        U_display_grid = self._create_display_grid(U_display_corners)
        xx, yy, zz = U_display_grid[..., 0], U_display_grid[..., 1], U_display_grid[..., 2]

        r = np.sqrt(xx**2 + yy**2 + zz**2)
        display_theta = np.arctan2(np.sqrt(xx**2 + yy**2), zz)
        display_phi = np.arctan2(zz, xx)
        return display_theta, display_phi

    '''frame 그리드 생성'''
    def _get_frame_grid(self, frame):
        frame_height, frame_width, _ = frame.shape
        xx, yy = np.meshgrid(np.linspace(-1, 1, frame_width), np.linspace(-1, 1, frame_height))
        frame_theta = xx * self.PI
        frame_phi = yy * self.PI_2
        return frame_theta, frame_phi

    '''USAFoV 추출'''
    def toUSAFoV(self, frame, eye_center, rz):
        D_user_position = self._calculate_position(eye_center, rz)
        U_display_corners = self._calculate_corners(D_user_position)

        display_theta, display_phi = self._convert_to_spherical(U_display_corners)
        frame_theta, frame_phi = self._get_frame_grid(frame)

        result_image = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        for i in range(self.display_height):
            for j in range(self.display_width):
                index = np.argmin((frame_theta - display_theta[i, j])**2 + (frame_phi - display_phi[i, j])**2)
                y, x = divmod(index, frame.shape[1])
                result_image[i, j] = frame[y, x]

        map_x = np.interp(display_theta, frame_theta[0], np.arange(self.display_width))
        map_y = np.interp(display_phi, frame_phi[:, 0], np.arange(self.display_height))
        
        map_x, map_y = np.meshgrid(map_x, map_y)
        
        # 결과 이미지 생성을 위한 remap 사용
        result_image = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

        return result_image








"""/**MAIN 파일**/"""
from cv2 import cv2
from math import pi
import numpy as np
import math

'''사용자 정의값들'''
box_size = 300
half_size = box_size / 2

distance_R = 100
base_distance_cm = 30
base_width_px = 200

# 웹캠 좌표 (display frame)
webcam_position = np.array([0, half_size, 0])

# 디스플레이 꼭짓점 좌표 (display frame)
display_corners = np.array([
    [half_size, half_size, half_size],
    [-half_size, half_size, half_size],
    [half_size, half_size, -half_size],
    [-half_size, half_size, -half_size]
])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\0604_black_win.mp4'
cap = cv2.VideoCapture(0) 
video = cv2.VideoCapture(video_path) 

nfov = USAFoV(height=800, width=1600, webcam_position=webcam_position, display_corners=display_corners)
detector = FaceLandmarkDetector()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


while True:
    ret, frame = video.read()
    if not ret:
        break

    success, image = cap.read()
    if not success:
        print("웹캠 오류")
        break

    results, image = detector.process_frame(image)

    right_eye_points, left_eye_points = detector.draw_landmarks(image, results)
    if right_eye_points and left_eye_points:
        eye_center = detector.get_eye_center(right_eye_points, left_eye_points)
        _, face_size = detector.get_face_size(results, image.shape)
        face_width = face_size[0]
        px_to_cm = base_distance_cm / base_width_px
        distance_cm = face_width * px_to_cm
        rz = distance_cm  # 모니터와 사람 사이의 거리 (cm단위)
        
        frame_nfov = nfov.toUSAFoV(frame, eye_center, rz)
    else:
        frame_nfov = nfov.toUSAFoV(frame, eye_center, rz)

    cv2.imshow('360 View', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()