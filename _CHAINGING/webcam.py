import cv2
import numpy as np

cap = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(3)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("웹캠 1 오류")
        break

    success, frame2= cap2.read()
    if not success:
        print("웹캠 2 오류")
        break

    frame = cv2.flip(frame, 1)
    frame2 = cv2.flip(frame2, 1)

    cv2.imshow('cam1', frame)
    cv2.imshow('cam2', frame2)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
