# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:53:33 2024

@author: daphn
"""

import cv2

# Haarcascades dosyalarının yolu
face_cascade = cv2.CascadeClassifier("frontalface.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Webcam'den video akışını başlat
cap = cv2.VideoCapture(0)

while True:
    # Video akışından bir frame oku
    ret, frame = cap.read()

    # Gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Yüzün çevresine dikdörtgen çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Yüz bölgesini gri tonlamaya çevir
        roi_gray = gray[y:y+h, x:x+w]

        # Göz tespiti
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Gözün çevresine dikdörtgen çiz
            cv2.rectangle(frame[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Göz bölgesini al
            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]

            # Threshold işlemi
            _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)

            # Kontur hesapla
            contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame[y:y+h, x:x+w], contours, -1, (0, 0, 255), 2)

    # Frame'i göster
    cv2.imshow('Frame', frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('i'):
        break

# Video akışını serbest bırak
cap.release()

# Pencereyi kapat
cv2.destroyAllWindows()
