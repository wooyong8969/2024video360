"""/**LatLong Image에서 사용자가 정의한 중심점을 기준으로 NFOV 이미지 생성**/"""

'''
The MIT License (MIT)
Copyright (c) 2021 Vít Ambrož
https://github.com/VitaAmbroz/360Tracking

'''
from cv2 import cv2
from math import pi
import numpy as np

# NFOV 클래스를 정의
class NFOV():
    def __init__(self, height=400, width=800):
        # FOV(Field Of View)의 기본값[90°, 45°]으로 설정
        self.FOV = [0.5, 0.5]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.height = height
        self.width = width

        # 화면에 표시될 점들의 배열을 생성
        self.screen_points = self._get_screen_img()
        # 변환 과정에서 사용되는 여러 배열들 초기화
        self.convertedScreenCoord = None
        self.sphericalCoord = None
        self.sphericalCoordReshaped = None

    def _get_coord_rad_point(self, point):
        # 주어진 포인트는 정규화된 좌표이므로, 이를 실제 각도인 라디안으로 변환
        # 이 때, π와 π/2를 사용하여 x축과 y축의 범위를 [-π, π], [-π/2, π/2]로 설정
        return (point * 2 - 1) * np.array([self.PI, self.PI_2])

    def _get_coord_rad_screen_points(self):
        # 화면에 표시될 점들의 좌표를 실제 각도로 변환
        # (이를 통해 각 점의 위치를 구체적인 각도로 계산하여, 360도 이미지 내의 실제 위치를 찾을 수 있음)
        return (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        # 화면 크기에 맞게 NFOV 이미지 구성 시 기준이 될 점들의 그리드를 생성
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))  # 너비와 높이에 따른 2차원 배열 생성
        return np.array([xx.ravel(), yy.ravel()]).T  # 1차원 배열로 변환하여 반환

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        # 구면 좌표를 그노모닉(직각) 좌표로 변환
        # 360도 이미지의 특정 영역을 일반적인 2D 이미지로 투영하는 데 사용
        # 결과적으로 위도와 경도를 나타내는 값이 반환됨

        # 아래 링크를 통해 360도 이미지에서 NFOV 이미지로의 정확한 투영 방법을 이해할 수 있다함!
        # http://blog.nitishmutha.com/equirectangular/360degree/2017/06/12/How-to-project-Equirectangular-image-to-rectilinear-view.html
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


    def toNFOV(self, frame, center_point, computeRectPoints=False):
        # 입력된 360도 이미지(프레임)을 클래스 변수로 저장
        self.frame = frame
        # 이미지의 높이, 너비, 채널 수를 각각 변수에 저장
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        # 중심점을 라디안 단위로 변환하여 저장 -> NFOV 이미지 중심점 정하는 데 사용
        self.cp = self._get_coord_rad_point(point=center_point)

        # 화면에 표시될 점들을 실제 라디안 좌표로 변환 (변환된 스크린 좌표)
        self.convertedScreenCoord = self._get_coord_rad_screen_points()
        # 구면 좌표를 직각좌표로 변환하여, NFOV 이미지에서 표시될 점들을 계산
        self.sphericalCoord = self._calcSphericaltoGnomonic(self.convertedScreenCoord)

        # 구면 좌표를 이미지의 크기에 맞게 재배열 -> NFOV 이미지 생성에 사용
        self.sphericalCoordReshaped = self.sphericalCoord.reshape(self.height, self.width, 2).astype(np.float32) % 1

        # 계산된 좌표를 사용해 원본 이미지에서 NFOV 이미지 추출
        # cv2.remap 함수로 픽셀을 새로운 위치로 매핑
        out = cv2.remap(self.frame, (self.sphericalCoordReshaped[..., 0] * self.frame_width), (self.sphericalCoordReshaped[..., 1] * self.frame_height), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return out