"""
Author: Do Tieu Thien
Date: 25/1/2018
"""

"""-----------------------------------------------------------------------------
IMPORT 
-----------------------------------------------------------------------------"""
import cv2
import numpy as np
from func.get_images_and_ids import get_images_and_ids

"""-----------------------------------------------------------------------------
MAIN 
-----------------------------------------------------------------------------"""
def training():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces, ids = get_images_and_ids("dataset/")
    recognizer.train(faces, np.array(ids))
    recognizer.save('training/training.yml')
