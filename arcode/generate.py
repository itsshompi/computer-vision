# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2 as cv
import numpy as np

# Tamaño del aruco
marker_size = 100
# Tipo de aruco
marker_type = cv.aruco.DICT_4X4_50
# Id del aruco
marker_id = 1

aruco = cv.drawMarker(marker_type, marker_id, marker_size)
cv.imwrite("aruco.png", aruco)

print(cv.__version__)
# Load the predefined dictionary
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

marketType = 'DICT_4X4_50'

for i in range(4):
    # Generate the marker
    markerImage = np.zeros((200, 200), dtype=np.uint8)
    #markerImage = cv.aruco.drawMarker(dictionary, i, 200, markerImage, 1)
    markerImage = cv.drawMarker(markerImage, (0,0), (255,255,255), 1) 
     
    cv.imwrite("./markers/marker_"+marketType+"_"+str(i)+".png", markerImage)
