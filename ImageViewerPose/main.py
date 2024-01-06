import cv2
import os
import glob

import mediapipe as mp

# Crea los objetos de dibujo y pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# La carpeta que contiene las imágenes
folder_path = './Images'

# Obteniendo todos los nombres de archivo en la carpeta
image_files = glob.glob(os.path.join(folder_path, '*'))

# Índice para la imagen actual
idx = 0
flag = 0
plot = False
# Ciclo para mantener la ventana de imagen abierta
# Inicia la solución de pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        print("idx: ", idx)       
        if(flag == 0):
            # Leer la imagen actual   
            image = cv2.imread(image_files[idx])

            results = pose.process(image)
            # Dibuja las anotaciones de pose en la imagen
            image.flags.writeable = True
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            print(results)
            # Muestra la imagen
            cv2.imshow('Pose estimation', image)  
            if(plot):
                mp_drawing.plot_landmarks(results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            flag = 1  

        # Esperar una tecla
        key = cv2.waitKey(0)

        # Si se presiona la tecla 'n' o 'N'
        if key == ord('d') or key == ord('D'):
            # Avanzar al siguiente índice
            idx += 1

            # Si el índice es mayor que el número de archivos, volvemos al primer archivo
            if idx >= len(image_files):
                idx = 0

            flag = 0

        # Si se presiona la tecla 'b' o 'B'
        elif key == ord('a') or key == ord('A'):
            # Retroceder al índice anterior
            idx -= 1

            # Si el índice es menor que 0, volvemos al último archivo
            if idx < 0:
                idx = len(image_files) - 1
            flag = 0

        elif key == ord('q') or key == ord('Q'):
            idx -= 10

            while idx < 0:            
                idx = len(image_files) + idx
            print(idx)
            flag = 0

        elif key == ord('e') or key == ord('E'):
            idx += 10

            while idx >= len(image_files):            
                idx = idx - len(image_files)
            flag = 0

        elif key == ord('p') or key == ord('P'):
            plot = not plot
            flag = 0

        # Si se presiona la tecla 'q' o 'Q'
        elif key == 27:
            # Salir del ciclo
            break


# Cerrar todas las ventanas al finalizar
cv2.destroyAllWindows()