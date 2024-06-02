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

    '''이미지 처리'''
    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지를 BGR에서 RGB로 변환
        results = self.face_mesh.process(image_rgb)  # 얼굴 메시 처리 수행
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # 결과 이미지를 다시 BGR로 변환
        return results, image_bgr

    '''landmark 그리기'''
    def draw_landmarks(self, image, results):
        if results.multi_face_landmarks:  # 탐지된 얼굴 landmark가 있으면
            for face_landmarks in results.multi_face_landmarks:
                # 얼굴의 landmark와 edges 그리기
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,  # 이미 정의된 스타일로 landmark 그리기
                    connection_drawing_spec=self.drawing_spec) # 이미 정의된 스타일로 edges 그리기

        return image

    '''얼굴의 경계 좌표 구하기'''
    def get_face_bounds(self, results, image_shape):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [landmark.x * image_shape[1] for landmark in face_landmarks.landmark]
                ys = [landmark.y * image_shape[0] for landmark in face_landmarks.landmark]
                min_x, max_x = int(min(xs)), int(max(xs))
                min_y, max_y = int(min(ys)), int(max(ys))
                return min_x, min_y, max_x, max_y
        return None

    '''얼굴 크기 구하기'''
    def get_face_size(self, min_x, min_y, max_x, max_y):
        width = max_x - min_x
        height = max_y - min_y
        return width, height




detector = FaceLandmarkDetector()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("웹캠 오류")
        break

    frame = cv2.flip(frame, 1)

    results, image = detector.process_frame(frame)
    image = detector.draw_landmarks(image, results)

    bounds = detector.get_face_bounds(results, frame.shape)
    if bounds:
        min_x, min_y, max_x, max_y = bounds
        width, height = detector.get_face_size(min_x, min_y, max_x, max_y)
        print(f"Face width: {width}px, Face height: {height}px")
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # 결과 이미지 출력
    cv2.imshow('Face Landmarks', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
