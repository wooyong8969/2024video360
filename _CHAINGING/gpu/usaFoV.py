import cupy as cp
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
        self.display_corners = cp.array(display_corners)

    '''디스플레이 좌표계 - 사용자의 위치 계산'''
    def _calculate_df_position(self, eye_center, ry, webcam_theta, webcam_alpha, state):
        reverse = -1 if state >= 3 else 1
        D_user_position = (
          reverse * ry * ((((-2 * cp.tan(webcam_theta/2) * eye_center[0]) / self.image_width) +  cp.tan(webcam_theta/2))),  
          -ry,
          ry * (((-2 * cp.tan(webcam_alpha/2) * eye_center[1]) / self.image_height) +  cp.tan(webcam_alpha/2))
        )
        print("webcam theta, alpha:", webcam_theta, webcam_alpha)
        
        return D_user_position

    '''사용자 좌표계 - 디스플레이의 네 모서리 좌표 계산'''
    def _calculate_uf_corners(self, user_position):
        D_display_corners = self.display_corners
        user_position = cp.array(user_position)
        U_display_corners = D_display_corners - user_position

        return U_display_corners
    
    '''디스플레이 그리드 생성'''
    def _create_display_grid(self, display_corners):
        top_left, top_right, bottom_left, bottom_right = display_corners

        print("top_left:", top_left)
        print("top_right:", top_right)
        print("bottom_left:", bottom_left)
        print("bottom_right:", bottom_right)

        t_values_width = cp.linspace(0, 1, self.display_width)
        t_values_height = cp.linspace(0, 1, self.display_height)
        t_width, t_height = cp.meshgrid(t_values_width, t_values_height)

        top_interpolation = (1 - t_width[:, :, None]) * top_left + t_width[:, :, None] * top_right
        bottom_interpolation = (1 - t_width[:, :, None]) * bottom_left + t_width[:, :, None] * bottom_right
        grid_points = (1 - t_height[:, :, None]) * top_interpolation + t_height[:, :, None] * bottom_interpolation
        
        return grid_points


    '''직각좌표를 구면 좌표로 변환'''
    def _convert_to_spherical(self, display_grid):
        xx, yy, zz = display_grid[..., 0], display_grid[..., 1], display_grid[..., 2]

        display_theta = cp.arctan2(xx, yy)
        display_phi = cp.arctan2(zz, cp.sqrt(xx**2 + yy**2))
    
        return display_theta, display_phi
    
    '''영상 좌표계 - 사용자의 위치 계산'''
    def _calculate_vf_position(self, user_position):
        V_user_position = cp.array([
            user_position[0],
            user_position[1] + self.display_distance,
            user_position[2]
        ])

        return V_user_position

    '''영상 좌표계 - 직선과 구의 교점 계산'''
    def _calculate_vf_sphere_intersections(self, V_display_grid, V_user_position):
        direction = V_display_grid - V_user_position
        
        a = cp.einsum('ijk,ijk->ij', direction, direction)
        b = 2 * cp.einsum('ijk,k->ij', direction, V_user_position)
        c = cp.dot(V_user_position, V_user_position) - self.sphere_radius**2

        discriminant = b**2 - 4 * a * c

        sqrt_discriminant = cp.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)

        t = cp.where(t1 > 0, t1, t2)
        t = cp.where(t > 0, t, cp.nan)

        intersection_points = V_user_position + t[..., None] * direction
        
        return intersection_points

    '''USAFoV 추출'''
    def toUSAFoV(self, frame, image_shape, eye_center, ry, state):
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]

        stream = cv2.cuda_Stream()
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame, stream)

        D_user_position = self._calculate_df_position(eye_center, ry, self.PI_2, self.PI_2/640*480, state)
        print("D_user_position:", D_user_position)
        print("---------------------------------------------------------")

        if state == 1:      # /**사용자 고정 모드**/
            U_display_corners = self._calculate_uf_corners(D_user_position) # 디스플레이 위치 재계산
            print("U_display_corners")
            print(U_display_corners)
            print("---------------------------------------------------------")
            
            U_display_grid = self._create_display_grid(U_display_corners)
            display_grid = U_display_grid

        elif state >= 2:    # /**디스플레이 고정 모드**/
            V_user_position = self._calculate_vf_position(D_user_position)  # 사용자 위치 재계산
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

        map_x = cp.asarray((self.PI + display_theta) / self.PI_2 * self.frame_width, dtype=cp.float32)
        map_y = cp.asarray((self.PI_2 - display_phi / self.PI_2/2) * self.frame_height, dtype=cp.float32)
        
        result_gpu = cv2.cuda.remap(gpu_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP, stream=stream)
        result_image = result_gpu.download(stream)
        
        if state >= 3:
            result_image = cv2.flip(result_image, 1)
        
        return result_image
    