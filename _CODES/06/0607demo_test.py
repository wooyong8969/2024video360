'''/**dx, dy, dz 모두 반영한 기존 NFoV**/''' 
import cv2
import numpy as np
from face_landmark2 import FaceLandmarkDetector
from nfov2 import NFOV

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\0604_black_win.mp4'
cap = cv2.VideoCapture(0) 
video = cv2.VideoCapture(video_path) 

nfov = NFOV(height=400, width=800) # 크기 키울 시, 화질 깨짐, 비율 깨짐의 문제 존재
detector = FaceLandmarkDetector()

def calculate_dx(eye_center, frame_width):
    # 화면 중심과 눈 중심의 차이를 구해 dx로 사용
    screen_center = frame_width / 2
    dx = (eye_center[0] - screen_center) / frame_width
    return dx

def calculate_dy(distance_R=100, base_distance_cm=30, base_width_px=200):
    bounds = detector.get_face_bounds(results, frame.shape)
    if bounds:
        min_x, min_y, max_x, max_y = bounds
        face_width, face_height = detector.get_face_size(min_x, min_y, max_x, max_y)
    
    dy = (base_distance_cm * base_width_px) / face_width # 모니터와 사람 사이의 거리 (cm단위)
    #print(distance_cm)
    return dy

def calculate_dz(eye_center, frame_height):
    screen_center = frame_height / 2
    dz = (eye_center[1] - screen_center) / frame_height
    return dz

def calculate_da(eye_center, frame_width, frame_height):
    dx = calculate_dx(eye_center, frame_width)
    dz = calculate_dz(eye_center, frame_height)
    dy = calculate_dy()
    da = (dx / dy, dz / dy) * 10
    return da

while True:
    ret, frame = video.read()
    if not ret:
        print("영상 오류")
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
        da = calculate_da(eye_center, image.shape[1], image.shape[0])
        
        center_point = np.array([0.5 - da[0], 0.5 - da[1]])
        frame_nfov = nfov.toNFOV(frame, center_point)

        # 눈 랜드마크 출력
        for point in right_eye_points + left_eye_points:
            flipped_x = frame_nfov.shape[1] - int(point[0])
            cv2.circle(frame_nfov, (flipped_x, int(point[1])), 2, (0, 255, 0), -1)
    else:
        frame_nfov = nfov.toNFOV(frame, center_point)
        #frame_nfov = nfov.toNFOV(frame, np.array([0, 0]))

    frame_nfov = cv2.flip(frame_nfov, 1)
    cv2.imshow('360 View', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
