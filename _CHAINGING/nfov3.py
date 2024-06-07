import numpy as np
import cv2
from math import pi
import mediapipe as mp

class FaceLandmarkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.RIGHT_EYE_INDEX = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 476, 473, 474]
        self.LEFT_EYE_INDEX = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 471, 468, 469]

    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return results, image_bgr

    def draw_landmarks(self, image, results):
        right_eye_points = []
        left_eye_points = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx in self.RIGHT_EYE_INDEX:
                    x, y = int(face_landmarks.landmark[idx].x * image.shape[1]), int(face_landmarks.landmark[idx].y * image.shape[0])
                    right_eye_points.append((x, y))
                    cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
                for idx in self.LEFT_EYE_INDEX:
                    x, y = int(face_landmarks.landmark[idx].x * image.shape[1]), int(face_landmarks.landmark[idx].y * image.shape[0])
                    left_eye_points.append((x, y))
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)

        return right_eye_points, left_eye_points
    
    def get_eye_center(self, right_eye_points, left_eye_points):
        right_eye_center = np.mean(right_eye_points, axis=0)
        left_eye_center = np.mean(left_eye_points, axis=0)
        eye_center = (right_eye_center + left_eye_center) / 2
        return eye_center
    
    def get_face_bounds(self, results, image_shape):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [landmark.x * image_shape[1] for landmark in face_landmarks.landmark]
                ys = [landmark.y * image_shape[0] for landmark in face_landmarks.landmark]
                min_x, max_x = int(min(xs)), int(max(xs))
                min_y, max_y = int(min(ys)), int(max(ys))
                return min_x, min_y, max_x, max_y
        return None

    def get_face_size(self, min_x, min_y, max_x, max_y):
        width = max_x - min_x
        height = max_y - min_y
        return width, height








class IntersectionCalculator:
    def __init__(self, R, box_height, box_width):
        self.R = R
        self.half_height = box_height / 2
        self.half_width = box_width / 2
        self.window_corners = np.array([
            [self.half_width, self.half_height, 0],
            [-self.half_width, self.half_height, 0],
            [self.half_width, -self.half_height, 0],
            [-self.half_width, -self.half_height, 0]
        ])
    
    def calculate_intersections(self, dx, dy):
        position = np.array([dx, dy, 0])
        intersection_points = []
        for corner in self.window_corners:
            direction = corner - position
            norm_direction = direction / np.linalg.norm(direction)
            
            a = np.dot(norm_direction, norm_direction)
            b = 2 * np.dot(norm_direction, position)
            c = np.dot(position, position) - self.R**2

            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                t2 = (-b + np.sqrt(discriminant)) / (2 * a)
                if t1 > 0 and t2 > 0:
                    t = min(t1, t2)
                elif t1 > 0:
                    t = t1
                elif t2 > 0:
                    t = t2
                else:
                    continue
                intersection_point = position + t * norm_direction
                intersection_points.append(intersection_point)
        return intersection_points











class NFOV:
    def __init__(self, height, width, R):
        self.FOV = [0.5, 0.5]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.height = height
        self.width = width
        self.R = R

    def _get_coord_rad_point(self, point):
        return (point * 2 - 1) * np.array([self.PI, self.PI_2])

    def toNFOV(self, frame, intersections):
        frame_height, frame_width = frame.shape[:2]

        try:
            spherical_points = [self._get_coord_rad_point(intersection[:2] / self.R) for intersection in intersections]
            spherical_points = np.array(spherical_points)

            min_lon = spherical_points[:, 0].min()
            max_lon = spherical_points[:, 0].max()
            min_lat = spherical_points[:, 1].min()
            max_lat = spherical_points[:, 1].max()

            lon_range = max_lon - min_lon
            lat_range = max_lat - min_lat

            screen_x = ((spherical_points[:, 0] - min_lon) / lon_range) * self.width
            screen_y = ((spherical_points[:, 1] - min_lat) / lat_range) * self.height
            # /////////////////////////////


            projected_img = np.zeros((self.height, self.width, 3), dtype=frame.dtype)
            for i in range(self.height):
                for j in range(self.width):
                    lon = min_lon + j / self.width * lon_range
                    lat = min_lat + i / self.height * lat_range
                    x = int((lon / (2 * self.PI) + 0.5) * frame_width)
                    y = int((lat / self.PI + 0.5) * frame_height)
                    if 0 <= x < frame_width and 0 <= y < frame_height:
                        projected_img[i, j] = frame[y, x]

            return projected_img
        
        except Exception as e:
            print(f"Error in toNFOV: {e}")
            return frame









# 초기 설정
height, width, R = 400, 800, 300

calculator = IntersectionCalculator(R=R, box_height=height, box_width=width)
nfov = NFOV(height=height, width=width, R=R)
detector = FaceLandmarkDetector()

# 비디오 캡처 설정
video_path = r'D:\W00Y0NG\PRGM2\360WINDOW\2024video360\_video\0528_test_video.mp4'
cap = cv2.VideoCapture(0)
video = cv2.VideoCapture(video_path)

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
        intersections = calculator.calculate_intersections(dx, dy)
        frame_nfov = nfov.toNFOV(frame, intersections)
    else:
        frame_nfov = frame

    cv2.imshow('360 View', frame_nfov)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
