"""
Author: Do Tieu Thien
Date: 23/1/2018
"""

"""-----------------------------------------------------------------------------
IMPORT 
-----------------------------------------------------------------------------"""
import os
import numpy as np
from func.create_dataset import create_dataset
from func.training import training
from func.DetectFaces import DetectFaces

"""-----------------------------------------------------------------------------
MAIN 
-----------------------------------------------------------------------------"""
while(True):
    print("""Selecting mode:
    1. Creating new data 
    2. Training data  
    3. Detecting faces """)

    mode = int(input("Enter a number correspond with a mode: "))

    if mode == 1:
        print("Creating new data")
        num_users = [os.path.join("dataset", folder) for folder in os.listdir("dataset")]
        id = np.shape(num_users)[0] + 1
        create_dataset(id)

    elif mode == 2:
        print("Training")
        training()

    elif mode == 3:
        print("Detecting faces")
        DetectFaces()

