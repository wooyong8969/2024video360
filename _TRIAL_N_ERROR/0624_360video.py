import cv2
import numpy as np
import os

def screen_to_spherical(screen_x, screen_y, screen_width, screen_height):
    """ 화면 좌표를 구면 좌표로 변환 """
    fov = 90  # 시야각(Field of View)
    f = screen_height / (2 * np.tan(np.deg2rad(fov) / 2))
    
    x = (screen_x - screen_width / 2) / f
    y = (screen_y - screen_height / 2) / f
    z = 1.0
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # 위도
    phi = np.arctan2(y, x)  # 경도
    
    return np.rad2deg(theta), np.rad2deg(phi)

def get_latlong_image_value(lat, long, latlong_image):
    """ 구면 좌표에서 latlong 이미지의 값을 가져오기 """
    h, w, _ = latlong_image.shape
    
    # 위도와 경도를 이미지 좌표로 변환
    x = int((long + 180) / 360 * w)
    y = int((90 - lat) / 180 * h)
    
    return latlong_image[y % h, x % w]

def map_points_to_latlong_video(video_path, screen_width, screen_height):
    """ 360도 영상을 2D 화면에 매핑 """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 화면용 빈 이미지 생성
        output_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        for screen_y in range(screen_height):
            for screen_x in range(screen_width):
                lat, long = screen_to_spherical(screen_x, screen_y, screen_width, screen_height)
                pixel_value = get_latlong_image_value(lat, long, frame)
                output_image[screen_y, screen_x] = pixel_value
        
        cv2.imshow('Mapped Video', output_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 360도 영상 파일 경로
video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_VIDEO\0528_test_video.mp4'

# 화면 크기 설정
screen_width = 800
screen_height = 400

# 네 개의 점을 latlong 이미지 값으로 매핑
map_points_to_latlong_video(video_path, screen_width, screen_height)
