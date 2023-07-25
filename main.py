import cv2
import numpy as np
from ultralytics import YOLO  # YOLO: You Only Look Once
import torch  # torch: ope source machine learning library to train neural networks

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  # 0: default camera

    model = YOLO('yolov8m.pt')  # yolov8m.pt: 640x640

    while True:
        ret, frame = cap.read()  # ret: return value, frame: image

        if not ret:
            print('Error')
            break

        results = model(frame, device='mps')
        results = results[0]
        bboxes = np.array(results.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results.boxes.cls.cpu(), dtype="int")

        for cls, bbox in zip(classes, bboxes):
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, str(cls), (x, y - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.imshow('Video', frame)
        key = cv2.waitKey(1)  # 1ms
        if key == ord('q'):
            break

    cap.release()  # release the capture
    cv2.destroyAllWindows()  # close all windows
