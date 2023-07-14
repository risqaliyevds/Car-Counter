import cv2
from ultralytics import YOLO
import cvzone
from Tracker.sort import *

cap = cv2.VideoCapture("Video/cars chilonzor.mp4")

model = YOLO("../yolo-weights/yolov8n.pt")

dict_cls = {}
with open("coco.txt", "r") as f:
    for line in f:
        (key, val) = line.split(": u'")
        dict_cls[int(key)] = val.strip("',\n")

mask = cv2.imread("Images/mask.png")
mask = cv2.resize(mask, (1280, 720))

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#Line
limits_up = [650, 500, 1250, 500]
limits_down = [650, 500, 100, 500]

#Total
total_count_up = {}
total_count_down = {}
count_up = 0
count_down = 0

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            bbox = int(x1), int(y1), int(w), int(h)
            conf = round(float(box.conf[0]), 2)
            cls = box.cls[0]

            if (dict_cls[int(cls) + 1] in ['car', 'truck', 'bus', 'motorcycle']):
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 0, 255), 2)
    cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 0, 255), 2)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=5, t=2, rt=2)
        cvzone.putTextRect(img, f"{dict_cls[int(cls) + 1]}", (max(0, x1), max(20, y1)), scale=1, thickness=1, offset=5)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)

        if (limits_up[0] < cx < limits_up[2]) and ((limits_up[1] - 15) < cy < (limits_up[1] + 15)):
            if id not in total_count_up:
                total_count_up[id] = count_up + 1
                cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 255, 0), 2)
                count_up += 1

        if (limits_down[2] < cx < limits_down[0]) and ((limits_down[1] - 15) < cy < (limits_down[1] + 15)):
            if id not in total_count_down:
                total_count_down[id] = count_down + 1
                cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 255, 0), 2)
                count_down += 1

        cvzone.putTextRect(img, f"Count UP: {count_up}", (1000, 50), scale=2, thickness=1, offset=5)
        cvzone.putTextRect(img, f"Count DOWN: {count_down}", (50, 50), scale=2, thickness=1, offset=5)

    cv2.imshow("Image", img)
    cv2.waitKey(0)