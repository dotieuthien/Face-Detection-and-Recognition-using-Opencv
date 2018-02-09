"""
Author: Do Tieu Thien
Date: 31/1/2018
"""


"""-----------------------------------------------------------------------------
DESCRIPTION 
This function use data table which contains information of with and height of 
rectangle boundary correspond with distance from camera to face.
-----------------------------------------------------------------------------"""


"""-----------------------------------------------------------------------------
IMPORT 
-----------------------------------------------------------------------------"""
from xlrd import open_workbook
import numpy as np


"""-----------------------------------------------------------------------------
MAIN 
-----------------------------------------------------------------------------"""
n = 2
excel_file = open_workbook('data_distance.xlsx')
sheet = excel_file.sheet_by_index(0)
areas = []
distances = []
for i in range(n):
    area = sheet.cell(i+1, 0).value
    areas.append(area)

    distance = sheet.cell(i+1, 1).value
    distances.append(distance)
