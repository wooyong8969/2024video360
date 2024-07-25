import numpy as np
from math import pi, tan
import cv2
from time import time

class USAFoV():
    def __init__(self, display_shape, webcam_position, display_corners, display_distance, sphere_radius):
        self.PI = pi
        self.PI_2 = pi * 0.5

        self.frame = None
        self.display = None
        self.display_height = display_shape[0]
        self.display_width = display_shape[1]

        self.image_height = None
        self.image_width = None

        self.display_distance = display_distance

        self.sphere_radius = sphere_radius

        self.webcam_position = webcam_position
        self.display_corners = np.array(display_corners)

    '''디스플레이 좌표계 - 사용자의 위치 계산'''
    def _calculate_df_position(self, eye_center, ry, webcam_theta, webcam_alpha):
        D_user_position = (
          ry * ((((-2 * np.tan(webcam_theta/2) * eye_center[0]) / self.image_width) +  np.tan(webcam_theta/2))),  
          -ry,
          ry * (((-2 * np.tan(webcam_alpha/2) * eye_center[1]) / self.image_height) +  np.tan(webcam_alpha/2))
        )
        print("webcam theta, alpha", webcam_theta, webcam_alpha)
        
        return D_user_position

    '''사용자 좌표계 - 디스플레이의 네 모서리 좌표 계산'''
    def _calculate_uf_corners(self, user_position):
        D_display_corners = self.display_corners
        user_position = np.array(user_position)
        U_display_corners = D_display_corners - user_position

        return U_display_corners

    '''디스플레이 그리드 생성'''
    def _create_display_grid(self, display_corners):
        #A, B, C, D = self._calculate_plane_equation(self.display_corners)

        top_left = display_corners[0]
        bottom_right = display_corners[3]

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
    
    '''직각좌표를 구면 좌표로 변환'''
    def _convert_to_spherical(self, display_grid):
        xx, yy, zz = display_grid[..., 0], display_grid[..., 1], display_grid[..., 2]

        display_theta = np.arctan2(xx, yy)
        display_phi = np.arctan2(zz, np.sqrt(xx**2 + yy**2))
    
        return display_theta, display_phi
    
    '''영상 좌표계 - 사용자의 위치 계산'''
    def _calculate_vf_position(self, user_position):
        V_user_position = np.array([
            user_position[0],
            user_position[1] + self.display_distance,
            user_position[2]
        ])

        return V_user_position
    
    '''영상 좌표계 - 디스플레이의 네 모서리 좌표 계산'''
    def _calculate_vf_corners(self, display_corners):
        V__display_corners = display_corners.copy()
        V__display_corners[:, 1] += self.display_distance

        return V__display_corners
    
    '''영상 좌표계 - 직선과 구의 교점 계산'''
    def _calculate_vf_sphere_intersections(self, V_display_grid, V_user_position):
        intersections = []

        for point in V_display_grid.reshape(-1, 3): # 320000개의 행, 3개의 열(x, y, z)
            direction = point - V_user_position
            a = np.dot(direction, direction)
            b = 2 * np.dot(V_user_position, direction)
            c = np.dot(V_user_position, V_user_position) - self.sphere_radius**2

            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                t1 = (-b + np.sqrt(discriminant)) / (2 * a)
                t2 = (-b - np.sqrt(discriminant)) / (2 * a)
                
                # 디스플레이 쪽 벡터의 교점 선택
                t = t1 if t1 > 0 else t2  # 둘 중 양수인 값 선택

                if t > 0:
                    intersection_point = V_user_position + t * direction
                    intersections.append(intersection_point)

        intersections_array = np.array(intersections)
        intersections_array = intersections_array.reshape(V_display_grid.shape)
        return intersections_array

    
    '''USAFoV 추출'''
    def toUSAFoV(self, frame, image_shape, eye_center, ry, state):
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]

        print("image height, width", image_shape[0], image_shape[1])
        print("frame height, width", self.frame_height, self.frame_width)
        print("---------------------------------------------------------")

        D_user_position = self._calculate_df_position(eye_center, ry, self.PI_2, self.PI_2/640*480)
        print("D_user_position:", D_user_position)
        print("---------------------------------------------------------")


        if state == 1:      # /**사용자 고정 모드**/
            U_display_corners = self._calculate_uf_corners(D_user_position) # 디스플레이 위치 재계산
            print("U_display_corners")
            print(U_display_corners)
            print("---------------------------------------------------------")
            
            U_display_grid = self._create_display_grid(U_display_corners)
            display_grid = U_display_grid

            
        elif state == 2:    # /**디스플레이 고정 모드**/
            V_user_position = self._calculate_vf_position(D_user_position)  # 사용자 위치 재계산
            print("V_user_position:", V_user_position)
            print("---------------------------------------------------------")

            V_display_corners = self._calculate_vf_corners(self.display_corners)
            print("display_corners")
            print(self.display_corners)
            print("---------------------------------------------------------")

            V_display_grid = self._create_display_grid(V_display_corners)
            print("V_display_grid")
            print(V_display_grid)
            print("---------------------------------------------------------")

            V_view_grid = self._calculate_vf_sphere_intersections(V_display_grid, V_user_position)
            display_grid = V_view_grid


        else:               # /**예외처리**/
            print("state 오류. state:", state)


        display_theta, display_phi =  self._convert_to_spherical(display_grid)

        print("theta")
        print(display_theta)
        print("---------------------------------------------------------")

        print("phi")
        print(display_phi)
        print("---------------------------------------------------------")

        result_image = cv2.remap(frame, (((self.PI + display_theta) / self.PI/2) * self.frame_width).astype(np.float32), ((self.PI_2 - display_phi / self.PI_2/2) * self.frame_height).astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)



        return result_image



        
        