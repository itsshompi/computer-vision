import mediapipe as mp
import cv2
import time
import struct
import socket
import time
import argparse


def showcams():
    # Loop over camera indexes until no more cameras are found
    print("Searching available cameras...")
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        print(f"Camera Key:: {index}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        cap.release()
        index += 1  

def main():  
    # If the showcams argument is provided, call the showcams function and exit
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--port', type=int, default=37020, help='UDP port')
    parser.add_argument('--instance', type=int, default=100, help='Instance ID')
    parser.add_argument('--flip', action='store_true', help='Flip the image horizontally')
    parser.add_argument('--showcams', action='store_true', help='Show available cameras')
    args = parser.parse_args()

    # Set the camera, port, instance, and flip variables
    CAMERA = args.camera
    PORT = args.port
    INSTANCE = args.instance
    FLIP = args.flip
    FRAME_INDEX = 0

    if args.showcams:
        showcams()
        return
    
    print("Starting Script with the following arguments...")
    print(f"Camera: {CAMERA}")
    print(f"Port: {PORT}")
    print(f"Instance: {INSTANCE}")
    print(f"Flip: {FLIP}")

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    server.settimeout(0.2)
    server.bind(("", 44444))


    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5)

    cap = cv2.VideoCapture(CAMERA)

    # Initialize the FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if FLIP:
            image = cv2.flip(image, 1)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            data = struct.pack('h', INSTANCE)
            data += struct.pack('I', FRAME_INDEX)
            for i in range(len(results.multi_hand_landmarks)):
                hand_landmarks = results.multi_hand_landmarks[i]
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Send the hand landmarks over UDP            
                boolean_value = 1 if results.multi_handedness[i].classification[0].label == 'Right' else 0
                print(boolean_value)
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    data_byte = idx & 0x7F
                    if boolean_value:
                        data_byte |= 0x80
                    data += struct.pack('B', data_byte)
                    data += struct.pack('2f', landmark.x , landmark.y)
            server.sendto(data, ('<broadcast>', PORT))
        
        cv2.imshow('MediaPipe Hands', image)

        # Calculate the FPS
        fps_frame_count += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_frame_count / (time.time() - fps_start_time)
            print('FPS:', fps)
            fps_frame_count = 0
            fps_start_time = time.time()

        key = cv2.waitKey(0)
        
        if key == 27:
            # Salir del ciclo
            break

        FRAME_INDEX += 1
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()