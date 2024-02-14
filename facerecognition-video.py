# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:55:43 2024

@author: daphn
"""

import cv2
import imutils
import face_recognition
 

color = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX


cap = cv2.VideoCapture(0)

pathTaylor = "profil.jpg" 
taylorImage = face_recognition.load_image_file(pathTaylor)
taylorImageEncodings = face_recognition.face_encodings(taylorImage)[0]


encodingsList = [taylorImageEncodings]
namesList = ["Defne"]


while True:
    
    ret,frame = cap.read()
    
    if ret == False:
        break
    
    rows, columns, channels = frame.shape
    
    coefficient = 4
    currentColumn = int(columns/coefficient)
    
    frame = imutils.resize(frame, width = currentColumn)
    
    faceLocations = face_recognition.face_locations(frame)
    faceEncodings = face_recognition.face_encodings(frame, faceLocations)
    
    
    for faceLoc, faceEncoding in zip(faceLocations,faceEncodings):
        topLeftY,bottomRightX,bottomRightY,topLeftX = faceLoc
        matchedFaces = face_recognition.compare_faces(encodingsList, faceEncoding)
        
        name = "unknown"
        
        if True in matchedFaces:
            matchedIndex = matchedFaces.index(True)
            name = namesList[matchedIndex]
            
        cv2.rectangle(frame, (topLeftX,topLeftY), (bottomRightX,bottomRightY), color, 1)
        cv2.putText(frame, name, (topLeftX,topLeftY), font, 1/(coefficient/1.5), color, 1)
        
        cv2.imshow("Face Recognition",frame)
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        

cap.release()
cv2.destroyAllWindows()