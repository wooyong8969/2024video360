"""/**LatLong Image에서 사용자가 정의한 중심점을 기준으로 USAFoV 이미지 생성**/"""

from cv2 import cv2
from math import pi
import numpy as np

class USAFoV():
    def __init__(self, height, width):
        self.PI = pi
        self.PI_2 = pi * 0.5

        self.frame = None
        self.display = None
        self.display_height = height
        self.display_width = width

    '''웹캠 원점 - 사용자의 위치 계산'''
    def _calculate_position(self):
        pass


    '''사용자 원점 - 디스플레이의 네 모서리 좌표 계산'''
    def _calculate_corners(self):
        pass

    """
    def calculate_r_position(frame_width, frame_height):
    bounds = detector.get_face_bounds(results, frame.shape)
    if bounds:
        min_x, min_y, max_x, max_y = bounds
        face_width, face_height = detector.get_face_size(min_x, min_y, max_x, max_y)
    
    px_to_cm = base_distance_cm / base_width_px
    distance_cm = face_width * px_to_cm # 모니터와 사람 사이의 거리 (cm단위)
    rz = distance_cm

    screen_center = frame_width / 2
    rx = (eye_center[0] - screen_center) / frame_width

    screen_center = frame_height / 2
    ry = (eye_center[1] - screen_center) / frame_height

    return np.array([rx, ry, rz])
    
    """

    '''디스플레이 그리드의 직각좌표를 구면 좌표로 변환'''
    def _convert_to_spherical(self):
        xx, yy, zz = np.meshgrid(np.linspace(-self.display_width / 2, self.display_width / 2, self.display_width),
                             np.linspace(-self.display_height / 2, self.display_height / 2, self.display_height))
        zz = np.zeros_like(xx)  # zz는 모든 점이 같은 z-평면에 있다고 가정

        r = np.sqrt(xx**2 + yy**2 + zz**2)
        display_theta = np.arcsin(yy / r)  # 경도
        display_phi = np.arctan2(zz, xx)   # 위도

        return display_theta, display_phi
    
    '''그리드 생성'''
    def _get_frame_grid(self, frame):
        frame_height, frame_width, _ = self.frame.shape
        xx, yy = np.meshgrid(np.linspace(-1, 1, frame_width), np.linspace(-1, 1, frame_height))
        frame_theta = xx * self.PI
        frame_phi = yy * self.PI_2
        return frame_theta, frame_phi
    
    '''USAFoV 추출'''
    def toUSAFoV(self, frame, user_position, window_corners):
        result_image = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        display_theta, display_phi = self._convert_to_spherical()
        frame_theta, frame_phi = self._get_frame_grid(frame)

        for i in range(self.display_height):
            for j in range(self.display_width):
                index = np.argmin((frame_theta - display_theta[i, j])**2 + (frame_phi - display_phi[i, j])**2)
                y, x = divmod(index, self.frame.shape[1])
                result_image[i, j] = self.frame[y, x]

        return result_image


'''
user_position을 키보드 이용하여 조정할 수 있게 구현.

'''