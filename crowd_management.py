import cv2
import torch
import numpy as np


def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

cv2.namedWindow("FRAME")
cv2.setMouseCallback("FRAME", get_coordinates)


try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print("⚠️ Model couldn't load from hub. Trying local copy...")
    model = torch.hub.load('./yolov5', 'yolov5s', source='local', pretrained=True)


cap = cv2.VideoCapture('people.mp4')

if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()


area = [(306, 42), (263, 428), (967, 428), (785, 42)]


while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ End of video or cannot read frame.")
        break

    frame = cv2.resize(frame, (1020, 600))

    
    results = model(frame)
    detections = results.pandas().xyxy[0]

    people_inside = []
    for _, row in detections.iterrows():
        if row['name'] == 'person':
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            
            
            inside = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
            if inside >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                people_inside.append((cx, cy))

   
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)

    
    count = len(people_inside)
    cv2.putText(frame, f"Count: {count}", (50, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    
    if count > 16:
        cv2.putText(frame, "⚠️ Over Crowded", (50, 90),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("FRAME", frame)

   
    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 Exiting...")
        break


cap.release()
cv2.destroyAllWindows()
