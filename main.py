import numpy as np
import ultralytics
import cv2
import cvzone
from sort import *

traker = Sort(max_age=25, min_hits=5)

model = ultralytics.YOLO("yolov8l.pt")
mask = cv2.imread('2.png')
video = cv2.VideoCapture('video.mp4')
limimLine1 = (50, 450)
limimLine2 = (1200, 450)
passedCars = []
while video.isOpened():
    ret, frame = video.read()
    frame_mask = cv2.bitwise_and(frame, mask)
    cv2.line(frame, limimLine1, limimLine2, (0, 0, 255), 2)

    cx, cy = 0, 0
    results = model.predict(frame_mask, classes=[2, 3, 5, 7], stream=True)
    detections = np.empty((0, 4))
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            w, h = x2 - x1, y2 - y1

            # cvzone.cornerRect(frame, (x1, y1, w, h), l=5, t=5)

            currentArray = np.array((x1, y1, x2, y2))
            detections = np.vstack((detections, currentArray))

    resultTracker = traker.update(detections)
    for results in resultTracker:
        x1, y1, x2, y2, id = results.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)
        if limimLine1[0] < cx < limimLine2[0] and limimLine1[1] - 30 < cy < limimLine2[1] + 5:
            if passedCars.count(id) == 0:
                cv2.line(frame, limimLine1, limimLine2, (0, 255, 0), 2)
                passedCars.append(id)

        # cvzone.putTextRect(frame, f'{id}', (int(x1), int(y1)))
    cvzone.putTextRect(frame, f'{len(passedCars)}', (50, 50))
    cv2.imshow('Look', frame)
    key = cv2.waitKey(20)

    if (key == ord('q')) or key == 27:
        break
