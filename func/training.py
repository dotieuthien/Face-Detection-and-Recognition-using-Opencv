"""
Author: Do Tieu Thien
Date: 25/1/2018
"""

"""-----------------------------------------------------------------------------
IMPORT 
-----------------------------------------------------------------------------"""
import cv2
import os
import numpy as np
from PIL import Image

"""-----------------------------------------------------------------------------
MAIN 
-----------------------------------------------------------------------------"""
recognizer = cv2.face_LBPHFaceRecognizer.create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    # os.path.listdir get name of file or folder and os.path.join get path of them
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    # create empth face list
    faceSamples = []

    # create empty ID list
    ids = []

    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # ignore if the file does not have jpg extension :
        # os.path.split(imagePath)[1] = user.id.sampleNum.jpg
        if(os.path.split(imagePath)[1].split(".")[3]!='jpg'):
            continue

        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')

        # now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage,'uint8')

        # getting the Id from the image
        id = int(os.path.split(imagePath)[1].split(".")[1])

        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)

        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids


faces, ids = getImagesAndLabels("dataset")
recognizer.train(faces, np.array(ids))
recognizer.save('training/training.yml')