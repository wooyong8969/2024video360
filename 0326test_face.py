'''/**실시간으로 웹캠을 읽어들여, 눈 관련 랜드마크만 표시**/'''
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

RIGHT_EYE_INDICES = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 476, 473, 474]
LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 471, 468, 469]

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠오류")
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    right_eye_points = []
    left_eye_points = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 오른쪽 눈 랜드마크 좌표 저장 및 그리기
            for idx in RIGHT_EYE_INDICES:
                x = int(face_landmarks.landmark[idx].x * image.shape[1])
                y = int(face_landmarks.landmark[idx].y * image.shape[0])
                right_eye_points.append((x, y))
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

            # 왼쪽 눈 랜드마크 좌표 저장 및 그리기
            for idx in LEFT_EYE_INDICES:
                x = int(face_landmarks.landmark[idx].x * image.shape[1])
                y = int(face_landmarks.landmark[idx].y * image.shape[0])
                left_eye_points.append((x, y))
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            '''
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=None)
            '''
            
    # 오른쪽과 왼쪽 눈 랜드마크 좌표 출력
    print("Right Eye:", right_eye_points)
    print("Left Eye:", left_eye_points)
    
    image = cv2.flip(image, 1)
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
