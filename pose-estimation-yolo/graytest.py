import cv2
from ultralytics import YOLO

#model = YOLO('yolov8n-pose.pt')
frame = cv2.imread('./images/man-walking.tif')
b, g, r = cv2.split(frame)
print(r)
print(b == g)
#results = model(frame, conf=0.7, max_det=1, boxes=False)
#annotated_frame = results[0].plot(boxes=False)
#cv2.imshow("YOLOv8 Inference", annotated_frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()