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
    def __init__(self, height=800, width=1600):
        self.FOV = [0.5, 0.5]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.height = height
        self.width = width

        self.screen_points = self._get_screen_img()
        self.convertedScreenCoord = None
        self.sphericalCoord = None
        self.sphericalCoordReshaped = None

    def _get_coord_rad_point(self, point):
        return (point * 2 - 1) * np.array([self.PI, self.PI_2])

    def _get_coord_rad_screen_points(self):
        return (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, 1600), np.linspace(0, 1, 800))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
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
        self.frame = frame
        print(frame.shape[0])
        print(frame.shape[1])
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        self.cp = self._get_coord_rad_point(point=center_point)

        self.convertedScreenCoord = self._get_coord_rad_screen_points()
        self.sphericalCoord = self._calcSphericaltoGnomonic(self.convertedScreenCoord)

        self.sphericalCoordReshaped = self.sphericalCoord.reshape(800, 1600, 2).astype(np.float32) % 1
        
        out = cv2.remap(self.frame,
                        (self.sphericalCoordReshaped[..., 0] * self.frame_width),
                        (self.sphericalCoordReshaped[..., 1] * self.frame_height),
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return out