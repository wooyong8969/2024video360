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
    def _calculate_position(self, eye_center):
        pass
        #return D_user_position

    '''사용자 원점 - 디스플레이의 네 모서리 좌표 계산'''
    def _calculate_corners(self, user_position):
        D_display_corners = self.display_corners
        U_display_corners = ''''''
        pass
        #return U_display_corners

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
    def toUSAFoV(self, frame, eye_center):
        D_user_position = self._calculate_position(eye_center)
        U_display_corners = self._calculate_corners(D_user_position)

        display_theta, display_phi = self._convert_to_spherical(U_display_corners)
        frame_theta, frame_phi = self._get_frame_grid(frame)

        result_image = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        for i in range(self.display_height):
            for j in range(self.display_width):
                index = np.argmin((frame_theta - display_theta[i, j])**2 + (frame_phi - display_phi[i, j])**2)
                y, x = divmod(index, frame.shape[1])
                result_image[i, j] = frame[y, x]

        return result_image


'''
user_position을 키보드 이용하여 조정할 수 있게 구현.

'''