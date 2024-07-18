import cv2
import numpy as np
import time
from com_face_landmark import FaceLandmarkDetector
from com_nfov import NFOV

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\0604_black_win.mp4'
cap = cv2.VideoCapture(0) 
video = cv2.VideoCapture(video_path) 

nfov = NFOV(height=400, width=800)
detector = FaceLandmarkDetector()

def calculate_dx(eye_center, frame_width):
    screen_center = frame_width / 2
    dx = (screen_center - eye_center[0]) / frame_width
    return dx / 10

def crop_center(image, scale=3):
    height, width = image.shape[:2]
    new_height = height // scale
    new_width = width // scale
    
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    
    cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]
    return cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

loop_counter = 0
start_time = time.time()

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
        
        for point in right_eye_points + left_eye_points:
            flipped_x = frame_nfov.shape[1] - int(point[0])
            cv2.circle(frame_nfov, (flipped_x, int(point[1])), 2, (0, 255, 0), -1)
        
        center_point = np.array([0.5 - dx, 0.5])
        frame_nfov = nfov.toNFOV(frame, center_point)
    else:
        frame_nfov = nfov.toNFOV(frame, np.array([0.5, 0.5]))

    #이미지의 중앙부만 크롭
    #frame_nfov = crop_center(frame_nfov)

    cv2.imshow('360 View', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()