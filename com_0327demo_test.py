'''/**양쪽 눈의 중앙이 화면의 중심에 오도록 360도 영상 회전**/''' 
import cv2
import numpy as np
from anon_face_landmark import FaceLandmarkDetector
from anon_nfov import NFOV

video_path = r'2024video360/yujinhong.mp4'
cap = cv2.VideoCapture(0) 
video = cv2.VideoCapture(video_path) 

nfov = NFOV(height=400, width=800) # 크기 키울 시, 화질 깨짐, 비율 깨짐의 문제 존재
detector = FaceLandmarkDetector()

while True:
    ret, frame = video.read()
    if not ret:
        break

    success, image = cap.read()
    if not success:
        print("웹캠 오류")
        break

    results, image = detector.process_frame(image)
    right_eye_points, left_eye_points = detector.draw_landmarks(image, results)
    if right_eye_points and left_eye_points:
        eye_center = detector.get_eye_center(right_eye_points, left_eye_points)
        center_point = eye_center / np.array([image.shape[1], image.shape[0]])
        frame_nfov = nfov.toNFOV(frame, center_point)
    else:
        frame_nfov = nfov.toNFOV(frame, np.array([0.5, 0.5]))

    cv2.imshow('360 View', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
