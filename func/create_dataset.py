"""
Author: Do Tieu Thien
Date: 21/1/2018
"""

"""-----------------------------------------------------------------------------
IMPORT 
-----------------------------------------------------------------------------"""
import cv2
import os

"""-----------------------------------------------------------------------------
MAIN 
-----------------------------------------------------------------------------"""
def create_dataset(id):
    camera = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haar_cascade/haarcascade_frontalface_default.xml")
    sampleNum = 0

    # create a folder for data of user
    os.makedirs("dataset/user_" + str(id))

    while (True):
        ret, image = camera.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # incrementing number of sample
            sampleNum = sampleNum + 1

            # saving the captured face in the dataset/user_id folder
            cv2.imwrite("dataset/user_" + str(id) + "/" + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow("User data", image)

        # wait for 100 miliseconds
        cv2.waitKey(100)

        # break if the sample number is more than 20
        if sampleNum > 30:
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Done")