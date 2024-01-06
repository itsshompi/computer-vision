# This code is written by Sunita Nayak at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:   python3 augmented_reality_with_aruco.py --image=test.jpg
#                  python3 augmented_reality_with_aruco.py --video=test.mp4

import cv2 as cv
import argparse
import sys
import os.path
import numpy as np
import time
import json
import requests
import qrcode



def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    # El orden es: superior izquierda, superior derecha, inferior derecha, inferior izquierda
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


url = "https://api.unnamed.cl/media/upload"

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)


winName = "Augmented Reality using Aruco markers in OpenCV"
hasFrame, frame = cap.read()
height, width = frame.shape[:2]
frame2 = np.zeros([height,width+100, 3],dtype=np.uint8)
prevTopLeftId = -1
colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255)]
corner = [3,2,1,0]
rotate = [cv.ROTATE_90_COUNTERCLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_CLOCKWISE]
done = False
while not done:
  try:
      # get frame from the video
    if(cv.waitKey(1) > 0):
      done = True
    hasFrame, frame = cap.read()
    height, width = frame.shape[:2]
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        cv.waitKey(3000)
        break
    #Load the dictionary that was used to generate the markers.
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    
    # Initialize the detector parameters using default values
    parameters =  cv.aruco.DetectorParameters()

    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    
    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

   


    if(len(markerIds) == 4):
      original_frame = frame.copy()
      topLeftId = 0
      distanceTopLeft = -1
      arr = []
      points = [0,0,0,0]
      for i in range(len(markerIds)):
        id = markerIds[i][0]
        pt = np.squeeze(markerCorners[i])[corner[id]]
        point = (int(pt[0]), int(pt[1]))
        points[id] = [id, point]
        cv.circle(frame, point, 5, colors[id], -1)
        distance = np.sqrt(pt[0]**2 + pt[1]**2)
        arr.append([id, distance])
        if distanceTopLeft == -1:
          topLeftId = id
          distanceTopLeft = distance
        else:
          if distance < distanceTopLeft:
            topLeftId = id
            distanceTopLeft = distance


      if(prevTopLeftId != topLeftId):
        #prevTopLeftId = topLeftId
        print(arr)
        print(topLeftId)
        print(points)
        for i in range(topLeftId):
          el = points.pop(0)
          points.append(el)
          print(points)
        x = points[0][1][0] #arriba izquierda
        y = points[0][1][1] #arriba izquierda
        w = points[3][1][0] - points[0][1][0] #arriba derecha - arriba izquierda
        h = points[1][1][1] - points[0][1][1] #abajo izquierda - arriba izquierda
        print(x,y,w,h)
        crop_img = original_frame[y:y+h, x:x+w]
        if(topLeftId > 0):
          crop_img = cv.rotate(crop_img, rotate[topLeftId-1])
        
        pts = np.array([
          [points[0][1][0], points[0][1][1]], 
          [points[1][1][0], points[1][1][1]], 
          [points[2][1][0], points[2][1][1]], 
          [points[3][1][0], points[3][1][1]]
          ], dtype = "float32")
        
        ordered_pts = order_points(pts)

        # Calcular la matriz de transformación de perspectiva y aplicarla
        if(topLeftId % 2 == 0):
          width = 1024
          height = 720
        else:
          width = 720
          height = 1024
        rect = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype = "float32")
        
        print(ordered_pts)
        print(rect)
        M = cv.getPerspectiveTransform(ordered_pts, rect)
        print(M)
        warped = cv.warpPerspective(frame, M, (width, height))
        
        if(topLeftId > 0):
          warped = cv.rotate(warped, rotate[topLeftId-1])

        x_offset=y_offset=50
        f_height, f_width = frame.shape[:2]
        frame2 = np.zeros([f_height,f_width+100, 3],dtype=np.uint8)
        #frame2[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img
        print(warped.shape)
        print(frame2.shape) 
        frame2[0:warped.shape[0], 0:warped.shape[1]] = warped
        
        cv.imshow("cropped", cv.hconcat([frame, frame2]))
        '''
        ruta_imagen = './images/' + str(time.time()) + '.jpg'
        cv.imwrite(ruta_imagen, warped)
        files = {'file': open(ruta_imagen, 'rb')}
        response = requests.post(url, files=files)

        print(response.text)
        if response.status_code == 201:
          print("Imagen subida con éxito.")
          qr = qrcode.QRCode(
              version=1,
              error_correction=qrcode.constants.ERROR_CORRECT_L,
              box_size=10,
              border=4,
          )
          data = json.loads(response.text)
          url_qr = data['url']
          qr.add_data(url_qr)
          qr.make(fit=True)

          img = qr.make_image(fill_color="black", back_color="white")
          img.resize((100, 100))
          img.save("./images/qr.jpg")
          
          qr_image = cv.imread("./images/qr.jpg")
          print(qr_image.shape)
          cv.imshow('qr', qr_image)
          while(cv.waitKey(1) < 0):
            pass
          done = True
          cv.destroyAllWindows()

        else:
          print("Error al subir la imagen.")'''

        

    cv.imshow("cropped", cv.hconcat([frame, frame2]))
   # cv.imshow("AR using Aruco markers", crop_img)


  except Exception as inst:
    print(inst)
    cv.imshow("cropped", cv.hconcat([frame, frame2]))

cap.release()
cv.destroyAllWindows()