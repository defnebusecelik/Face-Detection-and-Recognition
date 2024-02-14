# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:45:21 2024

@author: daphn
"""

import cv2
import dlib
import numpy as np


cap = cv2.VideoCapture(0)


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    # Görüntüyü oku
    ret, frame = cap.read()

    # Gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Ağız ve göz koordinatlarını al
        mouth_points = landmarks.parts()[48:68]
        eye_points = landmarks.parts()[36:48]

      
        mouth_frame = cv2.convexHull(np.array([(p.x, p.y) for p in mouth_points], dtype=np.int32))
        eye_frame = cv2.convexHull(np.array([(p.x, p.y) for p in eye_points], dtype=np.int32))

        # Threshold uygula
        mouth_mask = np.zeros_like(gray)
        eye_mask = np.zeros_like(gray)
        cv2.drawContours(mouth_mask, [mouth_frame], 0, 255, -1)
        cv2.drawContours(eye_mask, [eye_frame], 0, 255, -1)

        # Beyaz bölgelerin alanlarını hesapla
        mouth_area = cv2.countNonZero(mouth_mask)
        eye_area = cv2.countNonZero(eye_mask)

        r
        print("Mouth Area:", mouth_area)
        print("Eye Area:", eye_area)

 
    cv2.imshow("Frame", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('i'):
        break


cap.release()
cv2.destroyAllWindows()
