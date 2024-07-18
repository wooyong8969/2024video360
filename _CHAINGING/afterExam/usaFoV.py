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
            ((2 * ry * np.tan(webcam_theta/2) * eye_center[1]) / self.image_width) - ry * np.tan(webcam_theta/2),  
            ry,
            ((2 * ry * np.tan(webcam_alpha/2) * eye_center[0]) / self.image_height) + ry * np.tan(webcam_alpha/2)
        )
        print("D_user_position:", D_user_position)
        
        return D_user_position

    '''사용자 원점 - 디스플레이의 네 모서리 좌표 계산'''
    def _calculate_corners(self, user_position):
        D_display_corners = self.display_corners
        user_position = np.array(user_position)
        U_display_corners = D_display_corners - user_position
        print("U_display_corners")
        print(U_display_corners)
        print("---------------")
        return U_display_corners

    '''사용자 원점 - 디스플레이 그리드 생성'''
    def _create_display_grid(self, U_display_corners):
        #A, B, C, D = self._calculate_plane_equation(self.display_corners)

        top_left = U_display_corners[0]
        bottom_right = U_display_corners[3]

        x1, y, z1 = top_left
        x2, _, z2 = bottom_right

        x_values = np.linspace(x1, x2, self.display_width)
        z_values = np.linspace(z1, z2, self.display_height)

        z_grid, x_grid = np.meshgrid(z_values, x_values, indexing='ij')

        y_grid = np.full((self.display_height, self.display_width), y)

        U_display_grid_shape = (self.display_height, self.display_width, 3)
        U_display_grid = np.zeros(U_display_grid_shape, dtype=float)
        U_display_grid[..., 0] = x_grid
        U_display_grid[..., 1] = y_grid
        U_display_grid[..., 2] = z_grid

        return U_display_grid
    

    '''사용자 원점 - 디스플레이 그리드의 직각좌표를 구면 좌표로 변환'''
    def _convert_to_spherical(self, U_display_corners):
        U_display_grid = self._create_display_grid(U_display_corners)
        xx, yy, zz = U_display_grid[..., 0], U_display_grid[..., 1], U_display_grid[..., 2]

        display_theta = np.arctan2(xx, yy)
        display_phi = np.arctan2(zz, np.sqrt(xx**2 + yy**2))
        print("bbb", display_theta.shape, display_phi.shape)

        return display_theta, display_phi

    '''frame 그리드 생성
    def _get_frame_grid(self, frame):
        frame_height, frame_width, _ = frame.shape
        xx, zz = np.meshgrid(np.linspace(-1, 1, frame_width), np.linspace(-1, 1, frame_height))

        frame_phi = zz * self.PI_2    # 경도
        frame_theta = xx * self.PI        # 위도

        return frame_theta, frame_phi'''

    '''USAFoV 추출'''
    def toUSAFoV(self, frame, image_shape, eye_center, ry):
        st = time()
        
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]

        D_user_position = self._calculate_position(eye_center, ry, self.PI_2, self.PI_2)
        U_display_corners = self._calculate_corners(D_user_position)

        display_theta, display_phi = self._convert_to_spherical(U_display_corners)

        print("theta")
        print(display_theta)

        print("phi")
        print(display_phi)

        result_image = cv2.remap(frame, ((display_theta / self.PI) * self.display_width).astype(np.float32), ((display_phi / self.PI_2) * self.display_height).astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        print("e")
        print
        ed = time()
        
        print(ed - st)


        return result_image