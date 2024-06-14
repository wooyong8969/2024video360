import cv2
import mediapipe as mp
import numpy as np

class FaceLandmarkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.RIGHT_EYE_INDEX = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 476, 473, 474]
        self.LEFT_EYE_INDEX = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 471, 468, 469]

    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return results, image_bgr

    def draw_eye_landmarks(self, image, results):
        right_eye_points = []
        left_eye_points = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx in self.RIGHT_EYE_INDEX:
                    x, y = int(face_landmarks.landmark[idx].x * image.shape[1]), int(face_landmarks.landmark[idx].y * image.shape[0])
                    right_eye_points.append((x, y))
                    cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

                for idx in self.LEFT_EYE_INDEX:
                    x, y = int(face_landmarks.landmark[idx].x * image.shape[1]), int(face_landmarks.landmark[idx].y * image.shape[0])
                    left_eye_points.append((x, y))
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        return right_eye_points, left_eye_points
    
    def get_eye_center(self, right_eye_points, left_eye_points):
        right_eye_center = np.mean(right_eye_points, axis=0)
        left_eye_center = np.mean(left_eye_points, axis=0)
        
        eye_center = (right_eye_center + left_eye_center) / 2
        return eye_center


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = FaceLandmarkDetector()

    cv2.namedWindow('Face Landmarks', cv2.WINDOW_NORMAL)
    
    screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    cv2.resizeWindow('Face Landmarks', int(screen_width), int(screen_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results, annotated_image = detector.process_frame(frame)
        right_eye_points, left_eye_points = detector.draw_eye_landmarks(annotated_image, results)

        if right_eye_points and left_eye_points:
            eye_center = detector.get_eye_center(right_eye_points, left_eye_points)
            print("Right eye points:", right_eye_points)
            print("Left eye points:", left_eye_points)
            print("Eye center:", eye_center)
            cv2.circle(annotated_image, tuple(eye_center.astype(int)), 3, (0, 255, 255), -1)
        annotated_image = cv2.flip(annotated_image, 1)
        cv2.imshow('Face Landmarks', annotated_image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
