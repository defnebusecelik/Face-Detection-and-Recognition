# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:33:53 2024

@author: daphn
"""

import cv2
import time
import face_recognition

time1 = time.time()

image = cv2.imread("lana.jpg")
faceLocs = face_recognition.face_locations(image,model="cnn")
color = (0,0,255)

for index,faceLoc in enumerate(faceLocs):
    topLeftY, bottomRightX, bottomRightY, topLeftX = faceLoc
    
    detectedFaces = image[topLeftY:bottomRightY, topLeftX:bottomRightX]
    
    cv2.imwrite("croppedFace.jpg",detectedFaces)
    cv2.rectangle(image, (topLeftX,topLeftY),(bottomRightX,bottomRightY),color,1)
    
    
    cv2.imshow("Cropped Face", detectedFaces)
    cv2.imshow("Test Image", image)
    cv2.waitKey()

time2 = time.time()

speedCNN = time2 - time1
print(speedCNN) #431.98464345932007