import cv2
import mediapipe as mp

# Crea los objetos de dibujo y pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Inicia la captura de vídeo
cap = cv2.VideoCapture(0)

# Inicia la solución de pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        # Convierte la imagen a RGB para MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesa la imagen y obtén los resultados de pose
        results = pose.process(image)
        # Dibuja las anotaciones de pose en la imagen
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Muestra la imagen
        cv2.imshow('Pose estimation', image)        
        # Termina el bucle si se presiona la tecla ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()