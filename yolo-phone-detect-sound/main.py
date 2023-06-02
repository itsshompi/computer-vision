import cv2
from ultralytics import YOLO
import time
from pydub import AudioSegment
from pydub.playback import play
import multiprocessing
import threading

alarm = AudioSegment.from_file('./audio/alarm.wav', format='wav')

def playAlarm():
    play(alarm)
    return


# Load the YOLOv8 model
model = YOLO('yolov8n.pt') 
names = model.names
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
audio_playing = False

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        start_time = time.time()
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.7, max_det=20)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print("FPS :", fps)       
        for r in results:
            for c in r.boxes.cls:
                if(names[int(c)] == 'cell phone' and not audio_playing):
                    audio_playing = True
                    t = threading.Thread(target=playAlarm)
                    t.start()
        try:
            if audio_playing and not t.is_alive():
                audio_playing = False
        except NameError:
            pass  
        
        cv2.putText(annotated_frame, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
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