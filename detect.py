"""
Author: Do Tieu Thien
Date: 22/1/2018
"""

"""-----------------------------------------------------------------------------
IMPORT 
-----------------------------------------------------------------------------"""
import cv2

"""-----------------------------------------------------------------------------
MAIN 
-----------------------------------------------------------------------------"""
recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read('training/training.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, image = camera.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(image,(x, y),(x+w, y+h),(225, 0, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if(conf > 50):
            if(id == 1):
                id = "Boss"
            if(id == 2):
                id = "Assistant"

            cv2.putText(image , id, (x, y+h), font, 1, (0, 255, 255), 2)

    cv2.imshow('Face',image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()