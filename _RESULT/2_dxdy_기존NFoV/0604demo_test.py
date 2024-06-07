'''/**앞 뒤 거리에 따라 dx 보정값 다르게 구현, 눈 랜드마크 동시 출력**/''' 
import cv2
import numpy as np
from face_landmark2 import FaceLandmarkDetector
from nfov2 import NFOV

video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\0604_black_win.mp4'
cap = cv2.VideoCapture(0) 
video = cv2.VideoCapture(video_path) 

nfov = NFOV(height=400, width=800) # 크기 키울 시, 화질 깨짐, 비율 깨짐의 문제 존재
detector = FaceLandmarkDetector()

# dx 보정값 계산
def calculate_k(distance_R=100, base_distance_cm=30, base_width_px=200):
    bounds = detector.get_face_bounds(results, frame.shape)
    if bounds:
        min_x, min_y, max_x, max_y = bounds
        face_width, face_height = detector.get_face_size(min_x, min_y, max_x, max_y)
    
    px_to_cm = base_distance_cm / base_width_px
    distance_cm = face_width * px_to_cm # 모니터와 사람 사이의 거리 (cm단위)

    k = 100 / distance_cm    # 해당 부분 합리적이지 X
    return k

def calculate_dx(eye_center, frame_width):
    # 화면 중심과 눈 중심의 차이를 구해 dx로 사용
    screen_center = frame_width / 2
    dx = (screen_center - eye_center[0]) / frame_width
    k = calculate_k()
    return dx / k

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
        
        # 원근 투영 변환을 위한 새로운 중심점 계산
        center_point = np.array([0.5 + dx, 0.5])
        frame_nfov = nfov.toNFOV(frame, center_point)

        # 눈 랜드마크 출력
        #for point in right_eye_points + left_eye_points:
        #    flipped_x = frame_nfov.shape[1] - int(point[0])
        #    cv2.circle(frame_nfov, (flipped_x, int(point[1])), 2, (0, 255, 0), -1)
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
