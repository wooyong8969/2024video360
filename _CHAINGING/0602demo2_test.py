'''/**dx, dy에 따라 시야각 계산하여 투영 시도**/''' 
import cv2
import numpy as np
from face_landmark2 import FaceLandmarkDetector
from ??????????? import NFOV

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_video\0528_test_video.mp4'
cap = cv2.VideoCapture(0) 
video = cv2.VideoCapture(video_path) 

nfov = NFOV(height=400, width=800) 
detector = FaceLandmarkDetector()

def calculate_dy(distance_R=100, base_distance_cm=30, base_width_px=200):
    bounds = detector.get_face_bounds(results, frame.shape)
    if bounds:
        min_x, min_y, max_x, max_y = bounds
        face_width, face_height = detector.get_face_size(min_x, min_y, max_x, max_y)
    
    px_to_cm = base_distance_cm / base_width_px
    distance_cm = face_width * px_to_cm # 모니터와 사람 사이의 거리 (cm단위)

    return distance_cm

def calculate_dx(eye_center, frame_width):
    # 화면 중심과 눈 중심의 차이를 구해 dx로 사용
    screen_center = frame_width / 2
    dx = (screen_center - eye_center[0]) / frame_width
    return dx

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    success, image = cap.read()
    if not success:
        print("웹캠 오류")
        break

    results, image = detector.process_frame(image)
    right_eye_points, left_eye_points = detector.draw_landmarks(image, results)
    if right_eye_points and left_eye_points:
        eye_center = detector.get_eye_center(right_eye_points, left_eye_points)
        dx = calculate_dx(eye_center, image.shape[1])
        dy = calculate_dy()
        position = np.array([dx, dy])
    else:
        

    cv2.imshow('360 View', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
