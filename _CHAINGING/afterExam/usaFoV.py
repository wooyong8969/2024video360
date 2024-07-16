import numpy as np
from math import pi, tan
import cv2
from time import time

class USAFoV():
    def __init__(self, display_shape, webcam_position, display_corners):
        self.PI = pi
        self.PI_2 = pi * 0.5

        self.frame = None
        self.display = None
        self.display_height = display_shape[0]
        self.display_width = display_shape[1]

        self.image_height = None
        self.image_width = None

        self.webcam_position = webcam_position
        self.display_corners = np.array(display_corners)

    '''디스플레이 중앙 원점 - 사용자의 위치 계산'''
    def _calculate_position(self, eye_center, ry, webcam_theta, webcam_alpha):
        D_user_position = (
            (2 * (eye_center[0] + self.image_width/2) * ry * np.tan(webcam_theta/2)) / self.display_width,  
            ry,
            (2 * (eye_center[1] + self.image_height/2)* ry * np.tan(webcam_alpha/2)) / self.display_height
        )
        return D_user_position

    '''사용자 원점 - 디스플레이의 네 모서리 좌표 계산'''
    def _calculate_corners(self, user_position):
        D_display_corners = self.display_corners
        user_position = np.array(user_position)
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
        A, B, C, D = self._calculate_plane_equation(self.display_corners)

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

        display_theta = np.arctan2(yy, xx)
        display_phi = np.arctan2(np.sqrt(xx**2 + yy**2), zz)

        return display_theta, display_phi

    '''frame 그리드 생성'''
    def _get_frame_grid(self, frame):
        frame_height, frame_width, _ = frame.shape
        xx, yy = np.meshgrid(np.linspace(-1, 1, frame_width), np.linspace(-1, 1, frame_height))

        frame_theta = yy * self.PI_2    # 경도
        frame_phi = xx * self.PI        # 위도

        return frame_theta, frame_phi

    '''USAFoV 추출'''
    def toUSAFoV(self, frame, image_shape, eye_center, ry):
        print("p")
        st = time()
        print("a")
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]

        D_user_position = self._calculate_position(eye_center, ry, self.PI_2, self.PI*8/9)
        U_display_corners = self._calculate_corners(D_user_position)

        display_theta, display_phi = self._convert_to_spherical(U_display_corners)
        
        print("b", display_theta.shape, display_phi.shape)

        #frame_theta, frame_phi = self._get_frame_grid(frame)
        print("c",  frame.shape[0],  frame.shape[1])
        
        map_i = (((self.PI_2 - display_phi) / (self.PI)) * frame.shape[0]).T
        map_j = (((self.PI - display_theta) / (2 * self.PI)) * frame.shape[1]).T


        print("d", map_i.shape, map_j.shape)
        result_image = cv2.remap(frame, map_i.astype(np.float32), map_j.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        print("e")
        ed = time()
        
        print(ed - st)


        return result_image