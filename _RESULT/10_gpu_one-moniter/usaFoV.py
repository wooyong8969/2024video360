import cupy as cp  # numpy 대신 cupy를 사용
from math import pi, tan
import cv2
from time import time

class USAFoV():
    def __init__(self, display_shape, webcam_info, display_corners, sphere_radius):
        self.PI = pi
        self.PI_2 = pi * 0.5

        self.frame = None
        self.display = None
        self.display_height = display_shape[0]
        self.display_width = display_shape[1]

        self.image_height = None
        self.image_width = None

        self.sphere_radius = sphere_radius

        self.webcam_position = cp.array(webcam_info[0])
        self.horizon_tan = cp.array(webcam_info[1])
        self.vertical_tan = cp.array(webcam_info[2])

        self.display_corners = cp.array(display_corners)  # cupy 배열로 변경

    '''디스플레이 좌표계 - 사용자의 위치 계산'''
    def _calculate_df_position(self, eye_center, ry, webcam_theta, webcam_alpha, state):
        reverse = -1 if state >= 3 else 1
        # 각 계산 결과를 별도로 저장
        try:
            x_component = reverse * ry * ((((-2 * tan(webcam_theta / 2) * eye_center[0]) / self.image_width) + tan(webcam_theta / 2)))
            y_component = -ry
            z_component = ry * (((-2 * tan(webcam_alpha / 2) * eye_center[1]) / self.image_height) + tan(webcam_alpha / 2))

            # 각 값을 출력하여 디버깅
            # print(f"x_component: {x_component}")
            # print(f"y_component: {y_component}")
            # print(f"z_component: {z_component}")

            # CuPy 배열 생성
            D_user_position = cp.array([x_component, y_component, z_component], dtype=cp.float32)

            # print("webcam theta, alpha:", webcam_theta, webcam_alpha)
            return D_user_position

        except Exception as e:
            print(f"Error in _calculate_df_position: {e}")
            return None

    '''사용자 좌표계 - 디스플레이의 네 모서리 좌표 계산'''
    def _calculate_uf_corners(self, user_position):
        D_display_corners = self.display_corners
        user_position = cp.array(user_position)  # cupy 배열로 변경
        U_display_corners = D_display_corners - user_position

        return U_display_corners

    '''디스플레이 그리드 생성'''
    def _create_display_grid(self, display_corners):
        top_left = cp.array(display_corners[0])  # cupy 배열로 변경
        top_right = cp.array(display_corners[1])
        bottom_left = cp.array(display_corners[2])
        bottom_right = cp.array(display_corners[3])

        # print("top_left:", top_left)
        # print("top_right:", top_right)
        # print("bottom_left:", bottom_left)
        # print("bottom_right:", bottom_right)

        t_values_width = cp.linspace(0, 1, self.display_width)  # cupy.linspace 사용
        t_values_height = cp.linspace(0, 1, self.display_height)
        t_width, t_height = cp.meshgrid(t_values_width, t_values_height)  # cupy.meshgrid 사용

        # 상단과 하단의 내분점 계산
        top_interpolation = (1 - t_width[:, :, cp.newaxis]) * top_left + t_width[:, :, cp.newaxis] * top_right
        bottom_interpolation = (1 - t_width[:, :, cp.newaxis]) * bottom_left + t_width[:, :, cp.newaxis] * bottom_right

        # print("top_interpolation:", top_interpolation)
        # print("bottom_interpolation:", bottom_interpolation)

        # 최종 그리드 포인트 계산
        grid_points = (1 - t_height[:, :, cp.newaxis]) * top_interpolation + t_height[:, :, cp.newaxis] * bottom_interpolation

        # print("grid_points shape:", grid_points.shape)
        # print("grid_points sample:", grid_points[0, 0], grid_points[self.display_height // 2, self.display_width // 2], grid_points[-1, -1])

        return grid_points

    '''직각좌표를 구면 좌표로 변환'''
    def _convert_to_spherical(self, display_grid):
        xx, yy, zz = display_grid[..., 0], display_grid[..., 1], display_grid[..., 2]

        display_theta = cp.arctan2(xx, yy)  # cupy.arctan2 사용
        display_phi = cp.arctan2(zz, cp.sqrt(xx**2 + yy**2))  # cupy.sqrt 사용

        return display_theta, display_phi

    '''영상 좌표계 - 사용자의 위치 계산'''
    def _calculate_vf_position(self, user_position):
        # print("in")
        # print(f"user_position type: {type(user_position)}")
        # print(f"self.webcam_position type: {type(self.webcam_position)}")

        # CuPy 배열끼리 연산
        V_user_position = user_position + self.webcam_position

        # print("out")
        return V_user_position

    '''영상 좌표계 - 직선과 구의 교점 계산'''
    def _calculate_vf_sphere_intersections(self, V_display_grid, V_user_position):
        direction = V_display_grid - V_user_position

        a = cp.einsum('ijk,ijk->ij', direction, direction)  # cupy.einsum 사용
        b = 2 * cp.einsum('ijk,k->ij', direction, V_user_position)
        c = cp.dot(V_user_position, V_user_position) - self.sphere_radius**2  # cupy.dot 사용

        discriminant = b**2 - 4 * a * c

        sqrt_discriminant = cp.sqrt(discriminant)  # cupy.sqrt 사용
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)

        t = cp.where(t1 > 0, t1, t2)  # cupy.where 사용
        t = cp.where(t > 0, t, cp.nan)

        intersection_points = V_user_position + t[..., cp.newaxis] * direction

        return intersection_points
    
    def _calculate_vf_plane_intersections(self, V_display_grid, V_user_position):
        """영상 좌표계 - 직선과 평면의 교점 계산"""
        y_plane = cp.float32(360)
        direction = V_display_grid - V_user_position

        t = (y_plane - V_user_position[1]) / direction[..., 1]
        x_intersection = V_user_position[0] + t * direction[..., 0]
        z_intersection = V_user_position[2] + t * direction[..., 2]
        intersection_points = cp.array([x_intersection, cp.full_like(x_intersection, y_plane), z_intersection], dtype=cp.float32)
        return intersection_points

    '''USAFoV 추출'''
    def toUSAFoV(self, frame, image_shape, eye_center, ry, state):
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]

        W_user_position = self._calculate_df_position(eye_center, ry, self.PI/90*180, self.PI/45*180, state)
        print("W_user_position:", W_user_position)

        if state == 1:      # /**사용자 고정 모드**/
            U_display_corners = self._calculate_uf_corners(W_user_position)  # 디스플레이 위치 재계산
            U_display_grid = self._create_display_grid(U_display_corners)
            display_grid = U_display_grid

        elif state == 2:    # /**디스플레이 고정 모드**/
            V_user_position = self._calculate_vf_position(W_user_position)  # 사용자 위치 재계산
            V_display_grid = self._create_display_grid(self.display_corners)
            V_view_grid = self._calculate_vf_sphere_intersections(V_display_grid, V_user_position)
            display_grid = V_view_grid

        elif state in [3, 4]:        # /**거울, 투명모드**/
            V_user_position = self._calculate_vf_position(W_user_position)  # 사용자 위치 재계산
            V_display_grid = self._create_display_grid(self.display_corners)
            V_view_grid = self._calculate_vf_plane_intersections(V_display_grid, V_user_position)
            display_grid = V_view_grid

        else:               # /**예외처리**/
            print("state 오류. state:", state)

        if state in [3, 4]:  # remap 배열 생성
            x_map = (((display_grid[0] / (cp.float32(2) * self.horizon_tan)) + cp.float32(0.5)) * self.frame_width).astype(cp.float32).get()
            y_map = ((display_grid[2] / (self.vertical_tan * cp.float32(2))) * self.frame_height).astype(cp.float32).get()
        else:
            display_theta, display_phi = self._convert_to_spherical(display_grid)
            x_map = (((self.PI + display_theta) / self.PI/2) * self.frame_width).astype(cp.float32).get()
            y_map = ((self.PI_2 - display_phi / self.PI_2/2) * self.frame_height).astype(cp.float32).get()

        result_image = cv2.remap(frame, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


        if state >= 3:
            result_image = cv2.flip(result_image, 1)

        return result_image
