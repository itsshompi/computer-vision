# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2 as cv
import numpy as np
 
# Load the predefined dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
 

for i in range(4):
    # Generate the marker
    markerImage = np.zeros((200, 200), dtype=np.uint8)
    markerImage = cv.aruco.drawMarker(dictionary, i, 200, markerImage, 1);
     
    cv.imwrite("./markers/marker"+str(i)+".png", markerImage);
