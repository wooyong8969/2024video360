'''/**wasd에 따라 360영상 회전**/'''
from nfov import NFOV
import cv2
import numpy as np

# 비디오 경로 설정
video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\DownLoad\yujinhong.mp4'
video = cv2.VideoCapture(video_path)

# NFOV 인스턴스 생성
nfov = NFOV(height=400, width=800)

# 중심점 초기화 (정면을 가리킴)
center_point = np.array([0.5, 0.5])

while True:
    ret, frame = video.read()
    if not ret:
        break

    # NFOV 이미지 생성
    frame_nfov = nfov.toNFOV(frame, center_point)
    cv2.imshow('frame', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('w'):
        center_point[1] -= 0.01  # 위로 이동
    elif key == ord('s'):
        center_point[1] += 0.01  # 아래로 이동
    elif key == ord('a'):
        center_point[0] -= 0.01  # 왼쪽으로 이동
    elif key == ord('d'):
        center_point[0] += 0.01  # 오른쪽으로 이동

    center_point = np.clip(center_point, 0, 1)

video.release()
cv2.destroyAllWindows()