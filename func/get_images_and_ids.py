"""
Author: Do Tieu Thien
Date: 25/1/2018
"""

"""-----------------------------------------------------------------------------
IMPORT 
-----------------------------------------------------------------------------"""
import os
import cv2
import numpy as np
from PIL import Image

"""-----------------------------------------------------------------------------
MAIN 
-----------------------------------------------------------------------------"""
def get_images_and_ids(path):
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # get the path of all the files in the folder
    # os.path.listdir get name of file or folder and os.path.join get path of them
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    # create empth face list
    face_samples = []

    # create empty ID list
    ids = []

    # now looping through all the image paths and loading the Ids and the images
    for image_path in image_paths:
        # ignore if the file does not have jpg extension :
        # os.path.split(image_path)[2] = sampleNum.jpg
        if(os.path.split(image_path)[1].split(".")[1]!='jpg'):
            continue

        # loading the image and converting it to gray scale
        pilImage = Image.open(image_path).convert('L')

        # now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage,'uint8')

        # getting the Id from the image
        id = os.path.split(image_path)[0].split("/")[2]
        id = int(id.split("_")[1])

        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)

        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            face_samples.append(imageNp[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, ids
