# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:46:41 2024

@author: daphn
"""

import cv2
import face_recognition


pathTest = "taydel.jpg" 
color = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX


image = cv2.imread(pathTest)

pathTaylor = "taylor.jpg" 
taylorImage = face_recognition.load_image_file(pathTaylor)
taylorImageEncodings = face_recognition.face_encodings(taylorImage)[0]

pathLana = "lana.jpg"
lanaImage = face_recognition.load_image_file(pathLana)
lanaImageEncodings = face_recognition.face_encodings(lanaImage)[0]

encodingsList = [taylorImageEncodings, lanaImageEncodings]
namesList = ["Taylor Swift", "Lana Del Rey"]


testImage = face_recognition.load_image_file(pathTest)
faceLocations = face_recognition.face_locations(testImage)
faceEncodings = face_recognition.face_encodings(testImage, faceLocations)


for faceLoc, faceEncoding in zip(faceLocations,faceEncodings):
    topLeftY,bottomRightX,bottomRightY,topLeftX = faceLoc
    matchedFaces = face_recognition.compare_faces(encodingsList, faceEncoding)
    
    name = "unknown"
    
    if True in matchedFaces:
        matchedIndex = matchedFaces.index(True)
        name = namesList[matchedIndex]
        
    cv2.rectangle(image, (topLeftX,topLeftY), (bottomRightX,bottomRightY), color, 1)
    cv2.putText(image, name, (topLeftX,topLeftY), font, 1, color, 1)
    
    cv2.imshow("Face Recognition",image)
    cv2.waitKey()













