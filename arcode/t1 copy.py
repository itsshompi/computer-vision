# This code is written by Sunita Nayak at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:   python3 augmented_reality_with_aruco.py --image=test.jpg
#                  python3 augmented_reality_with_aruco.py --video=test.mp4

import cv2 as cv
import argparse
import sys
import os.path
import numpy as np
import time


cap = cv.VideoCapture(0, cv.CAP_DSHOW)

winName = "Augmented Reality using Aruco markers in OpenCV"
hasFrame, frame = cap.read()
height, width = frame.shape[:2]
frame2 = np.zeros([height,width+100, 3],dtype=np.uint8)

while cv.waitKey(1) < 0:
  try:
      # get frame from the video
    hasFrame, frame = cap.read()
    height, width = frame.shape[:2]
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        cv.waitKey(3000)
        break
    #Load the dictionary that was used to generate the markers.
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    
    # Initialize the detector parameters using default values
    parameters =  cv.aruco.DetectorParameters()

    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    
    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

    colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255)]
    
    index_0 = np.squeeze(np.where(markerIds==0))
    if len(index_0) > 0:
      pt_0 = np.squeeze(markerCorners[index_0[0]])[3]
      point_1 = (int(pt_0[0]), int(pt_0[1]))
      cv.circle(frame, point_1, 5, colors[0], -1)

    index_1 = np.squeeze(np.where(markerIds==1))
    if len(index_1) > 0:
      pt_1 = np.squeeze(markerCorners[index_1[0]])[0]
      point_2 = (int(pt_1[0]), int(pt_1[1]))
      cv.circle(frame, point_2, 5, colors[1], -1)
    
    index_2 = np.squeeze(np.where(markerIds==2))
    if len(index_2):
      pt_2 = np.squeeze(markerCorners[index_2[0]])[1]
      point_3 = (int(pt_2[0]), int(pt_2[1]))
      cv.circle(frame, point_3, 5, colors[2], -1)
    
    index_3 = np.squeeze(np.where(markerIds==3))
    if len(index_3):
      pt_3 = np.squeeze(markerCorners[index_3[0]])[2]
      point_4 = (int(pt_3[0]), int(pt_3[1]))
      cv.circle(frame, point_4, 5, colors[3], -1)

   

    
    

    if(markerIds[0][0] == 0 and len(markerIds[0])):
      
      print('marker 0')
      #Crop the image
      x = point_2[0] #arriba izquierda
      y = point_2[1] #arriba izquierda
      w = point_3[0] - point_2[0] #arriba derecha - arriba izquierda
      h = point_1[1] - point_2[1] #abajo izquierda - arriba izquierda

      if(w > 100):
        crop_img = frame[y:y+h, x:x+w] 
        crop_img = cv.rotate(crop_img, cv.ROTATE_90_CLOCKWISE)
        x_offset=y_offset=50
        frame2 = np.zeros([height,width+100, 3],dtype=np.uint8)
        frame2[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img
      
    if(markerIds[0][0] == 1 and len(markerIds[0])):
      print('marker 1')
      #Crop the image
      x = point_3[0]
      y = point_3[1]
      w = point_4[0] - point_3[0]
      h = point_2[1] - point_3[1]
      
      print(point_1)
      print(point_2)
      print(point_3)
      print(point_4)
      print(x,y,w,h)
      crop_img = frame[y:y+h, x:x+w]
      crop_img = cv.rotate(crop_img, cv.ROTATE_180)
      x_offset=y_offset=50
      frame2 = np.zeros([height,width+100, 3],dtype=np.uint8)
      frame2[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img
      #cv.imshow("cropped", crop_img)
    
    if(markerIds[0][0] == 2 and len(markerIds[0])):
      print('marker 2')
      #Crop the image
      x = point_4[0]
      y = point_4[1]
      w = point_2[0] - point_4[0]
      h = point_3[1] - point_4[1]

      crop_img = cv.rotate(frame[y:y+h, x:x+w], cv.ROTATE_180)
      x_offset=y_offset=50
      frame2 = np.zeros([height,width+100, 3],dtype=np.uint8)
      frame2[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img
      #cv.imshow("cropped", crop_img)

    if(markerIds[0][0] == 3 and len(markerIds[0])):

      print('marker 3')
      #Crop the image
      x = point_3[0]  
      y = point_3[1]
      w = point_1[0] - point_3[0]
      h = point_2[1] - point_3[1]
      crop_img = frame[y:y+h, x:x+w]
      cv.rotate(crop_img, cv.ROTATE_90_COUNTERCLOCKWISE)
      x_offset=y_offset=50
      frame2 = np.zeros([height,width+100, 3],dtype=np.uint8)
      frame2[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img
      #cv.imshow("cropped", crop_img)

    cv.imshow("cropped", cv.hconcat([frame, frame2]))
   # cv.imshow("AR using Aruco markers", crop_img)


  except Exception as inst:
    cv.imshow("cropped", cv.hconcat([frame, frame2]))

cv.destroyAllWindows()