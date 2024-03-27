'''/**wasd에 따라 조금 더 부드럽게 360영상 회전**/'''
import cv2
import numpy as np
from nfov import NFOV

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\DownLoad\yujinhong.mp4'
video = cv2.VideoCapture(video_path)
nfov = NFOV(height=400, width=800)
center_point = np.array([0.5, 0.5])

center_velocity = np.array([0.0, 0.0])
acceleration = 0.001
friction = 0.9

while True:
    ret, frame = video.read()
    if not ret:
        break

    center_velocity *= friction

    center_point += center_velocity
    center_point = np.clip(center_point, 0, 1)

    frame_nfov = nfov.toNFOV(frame, center_point)
    cv2.imshow('frame', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('w'):
        center_velocity[1] -= acceleration  # 위로 가속
    elif key == ord('s'):
        center_velocity[1] += acceleration  # 아래로 가속
    elif key == ord('a'):
        center_velocity[0] -= acceleration  # 왼쪽으로 가속
    elif key == ord('d'):
        center_velocity[0] += acceleration  # 오른쪽으로 가속

video.release()
cv2.destroyAllWindows()
