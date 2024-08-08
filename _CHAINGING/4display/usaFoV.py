import numpy as np
from math import pi, tan
import cv2
from time import time

class USAFoV():
    def __init__(self, display_shape, webcam_position, display_corners, sphere_radius):
        self.PI = pi
        self.PI_2 = pi * 0.5

        self.frame = None
        self.display = None
        self.display_height = display_shape[0]
        self.display_width = display_shape[1]

        self.image_height = None
        self.image_width = None

        self.sphere_radius = sphere_radius

        self.webcam_position = webcam_position
        self.display_corners = np.array(display_corners)

    '''디스플레이 좌표계 - 사용자의 위치 계산'''
    def _calculate_df_position(self, eye_center, ry, webcam_theta, webcam_alpha, state):
        reverse = -1 if state >= 3 else 1
        D_user_position = (
          reverse * ry * ((((-2 * np.tan(webcam_theta/2) * eye_center[0]) / self.image_width) +  np.tan(webcam_theta/2))),  
          -ry,
          ry * (((-2 * np.tan(webcam_alpha/2) * eye_center[1]) / self.image_height) +  np.tan(webcam_alpha/2))
        )
        print("webcam theta, alpha:", webcam_theta, webcam_alpha)
        
        return D_user_position

    '''사용자 좌표계 - 디스플레이의 네 모서리 좌표 계산'''
    def _calculate_uf_corners(self, user_position):
        D_display_corners = self.display_corners
        user_position = np.array(user_position)
        U_display_corners = D_display_corners - user_position

        return U_display_corners
    
    '''디스플레이 그리드 생성'''
    def _create_display_grid(self, display_corners):
        top_left = np.array(display_corners[0])
        top_right = np.array(display_corners[1])
        bottom_left = np.array(display_corners[2])
        bottom_right = np.array(display_corners[3])

        print("top_left:", top_left)
        print("top_right:", top_right)
        print("bottom_left:", bottom_left)
        print("bottom_right:", bottom_right)

        t_values_width = np.linspace(0, 1, self.display_width)
        t_values_height = np.linspace(0, 1, self.display_height)
        t_width, t_height = np.meshgrid(t_values_width, t_values_height)

        # 상단과 하단의 내분점 계산
        top_interpolation = (1 - t_width[:, :, np.newaxis]) * top_left + t_width[:, :, np.newaxis] * top_right
        bottom_interpolation = (1 - t_width[:, :, np.newaxis]) * bottom_left + t_width[:, :, np.newaxis] * bottom_right

        # 디버깅 출력
        print("top_interpolation:", top_interpolation)
        print("bottom_interpolation:", bottom_interpolation)

        # 최종 그리드 포인트 계산
        grid_points = (1 - t_height[:, :, np.newaxis]) * top_interpolation + t_height[:, :, np.newaxis] * bottom_interpolation

        # 디버깅 출력
        print("grid_points shape:", grid_points.shape)
        print("grid_points sample:", grid_points[0, 0], grid_points[self.display_height // 2, self.display_width // 2], grid_points[-1, -1])

        return grid_points


    '''직각좌표를 구면 좌표로 변환'''
    def _convert_to_spherical(self, display_grid):
        xx, yy, zz = display_grid[..., 0], display_grid[..., 1], display_grid[..., 2]

        display_theta = np.arctan2(xx, yy)
        display_phi = np.arctan2(zz, np.sqrt(xx**2 + yy**2))
    
        return display_theta, display_phi
    
    '''영상 좌표계 - 사용자의 위치 계산'''
    def _calculate_vf_position(self, user_position):
        V_user_position = np.array(user_position) + self.webcam_position
        return V_user_position

    '''영상 좌표계 - 직선과 구의 교점 계산'''
    def _calculate_vf_sphere_intersections(self, V_display_grid, V_user_position):
        direction = V_display_grid - V_user_position
    
        a = np.einsum('ijk,ijk->ij', direction, direction)
        b = 2 * np.einsum('ijk,k->ij', direction, V_user_position)
        c = np.dot(V_user_position, V_user_position) - self.sphere_radius**2

        discriminant = b**2 - 4 * a * c

        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)
        
        t = np.where(t1 > 0, t1, t2)
        t = np.where(t > 0, t, np.nan)
        
        intersection_points = V_user_position + t[..., np.newaxis] * direction
        
        return intersection_points

    '''USAFoV 추출'''
    def toUSAFoV(self, frame, image_shape, eye_center, ry, state):
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]

        #print("image height, width", image_shape[0], image_shape[1])
        #print("frame height, width", self.frame_height, self.frame_width)
        #print("---------------------------------------------------------")

        W_user_position = self._calculate_df_position(eye_center, ry, self.PI_2, self.PI_2/640*480, state)
        print("W_user_position:", W_user_position)
        print("---------------------------------------------------------")

        if state == 1:      # /**사용자 고정 모드**/
            U_display_corners = self._calculate_uf_corners(W_user_position) # 디스플레이 위치 재계산
            print("U_display_corners")
            print(U_display_corners)
            print("---------------------------------------------------------")
            
            U_display_grid = self._create_display_grid(U_display_corners)
            display_grid = U_display_grid

        elif state >= 2:    # /**디스플레이 고정 모드**/
            V_user_position = self._calculate_vf_position(W_user_position)  # 사용자 위치 재계산
            print("V_user_position:", V_user_position)
            print("---------------------------------------------------------")

            V_display_grid = self._create_display_grid(self.display_corners)
            print("V_display_grid")
            print(V_display_grid)
            print("---------------------------------------------------------")

            V_view_grid = self._calculate_vf_sphere_intersections(V_display_grid, V_user_position)
            display_grid = V_view_grid

        else:               # /**예외처리**/
            print("state 오류. state:", state)


        display_theta, display_phi =  self._convert_to_spherical(display_grid)

        # 거울모드, 투명모드에서 시야각 조정
        display_theta *= 5# if state >= 3 else 0
        display_phi *= 5 #if state >= 3 else 0

        #print("theta")
        #print(display_theta)
        #print("---------------------------------------------------------")

        #print("phi")
        #print(display_phi)
        #print("---------------------------------------------------------")

        result_image = cv2.remap(frame, (((self.PI + display_theta) / self.PI/2) * self.frame_width).astype(np.float32),
                                 ((self.PI_2 - display_phi / self.PI_2/2) * self.frame_height).astype(np.float32),
                                 interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        if state >= 3:
            result_image = cv2.flip(result_image, 1)
        
        return result_image
    