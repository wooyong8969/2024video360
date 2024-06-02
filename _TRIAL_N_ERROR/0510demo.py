import cv2
import numpy as np
from face_landmark import FaceLandmarkDetector
from nfov import NFOV

video_path = r'2024video360/yujinhong.mp4'
cap = cv2.VideoCapture(0) 
video = cv2.VideoCapture(video_path) 

nfov = NFOV(height=400, width=800)
detector = FaceLandmarkDetector()

def calculate_dx(eye_center, frame_width):
    screen_center = frame_width / 2
    dx = (eye_center[0] - screen_center) / frame_width
    return dx

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
        
        center_point = np.array([0.5 + dx, 0.5])
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
