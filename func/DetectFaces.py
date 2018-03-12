"""
Author: Do Tieu Thien
Date: 23/1/2018
"""

"""-----------------------------------------------------------------------------
IMPORT 
-----------------------------------------------------------------------------"""
import cv2

"""-----------------------------------------------------------------------------
MAIN 
-----------------------------------------------------------------------------"""
def DetectFaces():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("training/training.yml")
    faceCascade = cv2.CascadeClassifier("haar_cascade/haarcascade_frontalface_default.xml")
    camera = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, image = camera.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for(x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (225, 0, 0), 2)
            id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if(conf < 55):
                if id == 1:
                    id = "Thien"
                elif id == 3:
                    id = "Khoi"

            else:
                id = "Unknow"

            cv2.putText(image, str(id), (x, y+h), font, 1, (0, 255, 255), 2)

        cv2.imshow('Detecting faces', image)
        cv2.waitKey(25)
