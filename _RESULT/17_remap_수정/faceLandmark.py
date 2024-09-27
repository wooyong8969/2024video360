"""/**얼굴 landmark 탐지**/"""
import cv2
import mediapipe as mp 
import numpy as np 

class FaceLandmarkDetector:
    def __init__(self):
        # MediaPipe의 얼굴 mesh 관련 설정 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # 탐지할 수 있는 얼굴의 최대 개수
            refine_landmarks=True,  # 세부 landmark를 개선할 것인지 여부
            min_detection_confidence=0.5,  # 탐지 확신도의 최소값
            min_tracking_confidence=0.5)  # 추적 확신도의 최소값
        
        # landmark와 edges을 그리기 위한 설정 초기화
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # 오른쪽 눈과 왼쪽 눈의 landmark 인덱스 정의
        # 제 3자의 입장에서, 9시 방향부터 시계방향으로 / -3, -2, -1은 눈동자 좌표
        self.RIGHT_EYE_INDEX = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 476, 473, 474]
        self.LEFT_EYE_INDEX = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 471, 468, 469]

    '''이미지 처리'''
    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지를 BGR에서 RGB로 변환
        results = self.face_mesh.process(image_rgb)  # 얼굴 메시 처리 수행
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # 결과 이미지를 다시 BGR로 변환
        return results, image_bgr

    '''landmark 그리기'''
    def draw_landmarks(self, image, results):
        right_eye_points = []
        left_eye_points = []  # 각 landmark의 좌표를 저장할 리스트 생성

        if results.multi_face_landmarks:  # 탐지된 얼굴 landmark가 있으면
            for face_landmarks in results.multi_face_landmarks:
                # 오른쪽 눈 landmark 표시
                for idx in self.RIGHT_EYE_INDEX:
                    x, y = int(face_landmarks.landmark[idx].x * image.shape[1]), int(face_landmarks.landmark[idx].y * image.shape[0])
                    right_eye_points.append((x, y))
                    cv2.circle(image, (x, y), 3, (255, 0, 0), -1)  # 3크기의 속이 채워진 파란색 원 그리기
                # 왼쪽 눈 landmark 표시
                for idx in self.LEFT_EYE_INDEX:
                    x, y = int(face_landmarks.landmark[idx].x * image.shape[1]), int(face_landmarks.landmark[idx].y * image.shape[0])
                    left_eye_points.append((x, y))
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 3크기의 속이 채워진 초록색 원 그리기
                
                # 얼굴의 landmark와 edges 그리기
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,  # 이미 정의된 스타일로 landmark 그리기
                    connection_drawing_spec=self.drawing_spec) # 이미 정의된 스타일로 edges 그리기

        return right_eye_points, left_eye_points
    
    '''눈의 중앙 좌표 구하기'''
    def get_eye_center(self, right_eye_points, left_eye_points):
        # 각 눈 관련 좌표들의 평균값을 눈의 중앙 좌표라 가정
        right_eye_center = np.mean(right_eye_points, axis=0)
        left_eye_center = np.mean(left_eye_points, axis=0)
        
        eye_center = (right_eye_center + left_eye_center) / 2  # 두 눈의 좌표의 중앙 구하기
        return eye_center
    

    '''얼굴의 경계 좌표 및 크기 구하기'''
    def get_face_size(self, results, image_shape):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [landmark.x * image_shape[1] for landmark in face_landmarks.landmark]
                ys = [landmark.y * image_shape[0] for landmark in face_landmarks.landmark]
                min_x, max_x = int(min(xs)), int(max(xs))
                min_y, max_y = int(min(ys)), int(max(ys))
                width = max_x - min_x
                height = max_y - min_y
                return (min_x, min_y, max_x, max_y), (width, height)
        return None, None