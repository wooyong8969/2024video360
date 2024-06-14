'''/**양쪽 눈의 중앙이 화면의 중심에 오도록 360도 영상 회전**/''' 
import cv2
import numpy as np
from com_face_landmark import FaceLandmarkDetector
from com_nfov import NFOV

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\0528_test_video.mp4'
cap = cv2.VideoCapture(0) 
video = cv2.VideoCapture(video_path) 

nfov = NFOV(height=400, width=800)
detector = FaceLandmarkDetector()

def calculate_dx(eye_center, frame_width):
    # 화면 중심과 눈 중심의 차이를 구해 dx로 사용
    screen_center = frame_width / 2
    dx = (screen_center - eye_center[0]) / frame_width
    return dx / 15

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
        dx = calculate_dx(eye_center, image.shape[1])
        
        # 원근 투영 변환을 위한 새로운 중심점 계산
        center_point = np.array([0.5 - dx, 0.5])
        frame_nfov = nfov.toNFOV(frame, center_point)
    else:
        frame_nfov = nfov.toNFOV(frame, np.array([0.5, 0.5]))

    frame_nfov = cv2.resize(frame_nfov, (frame_nfov.shape[1] * 2, frame_nfov.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('360 View', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
