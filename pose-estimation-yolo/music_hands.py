import cv2
from ultralytics import YOLO
import time
from pydub import AudioSegment
from pydub.playback import play
import multiprocessing
import threading

audio1 = AudioSegment.from_file('./audio/audio1.wav', format='wav')
audio2 = AudioSegment.from_file('./audio/audio2.wav', format='wav')

def playFunc(n):
    if(n == 1):
        play(audio1)
    elif(n == 2):
        play(audio2)
    return


# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt') 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        start_time = time.time()
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.7, max_det=1, boxes=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot(boxes=False)

        # print keypoints index number and x,y coordinates
        #for idx, kpt in enumerate(results[0].keypoints[0]):
            #print(f"Keypoint {idx}: ({int(kpt[0])}, {int(kpt[1])})")
            #annotated_frame = cv2.putText(annotated_frame, f"{idx}:({int(kpt[0])}, {int(kpt[1])})", (int(kpt[0]), int(kpt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print("FPS :", fps)
        height, width, _ = annotated_frame.shape

        # Draw a horizontal line in the middle of the image
        start_point = (0, height // 2)
        end_point = (width, height // 2)
        color = (0, 0, 255)  # color of the line (BGR format)
        thickness = 2  # thickness of the line
        cv2.line(annotated_frame, start_point, end_point, color, thickness)

        try:
            right_hand = results[0].keypoints[0][10]
            left_hand = results[0].keypoints[0][9]
            p = right_hand
            cv2.circle(annotated_frame, (int(p[0]),int(p[1])), 20, (0, 0, 255), 2)
            print('left hand: ', left_hand[0], left_hand[1])
            if(int(left_hand[1]) < height // 2):
                threading.Thread(target=playFunc, args=(1,)).start()
            if(int(right_hand[1]) < height // 2):
                threading.Thread(target=playFunc, args=(2,)).start()
        except Exception as e:
            print(e)     
            print('error')       
            print(len(results[0].keypoints))
            pass
        
        cv2.putText(annotated_frame, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()