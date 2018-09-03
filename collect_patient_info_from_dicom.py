#Reference: https://github.com/smopart/lung_nodule/tree/master/project/code
#Modified by: Mandal, B.

import numpy as np
import pandas as pd
import pickle
import dicom
import os
import pdb

# Collecting Patient Information

FolderPath = "/home/asif/Desktop/CapstoneData/data7"

ArrayList = list()
IDs = list()
Ages = list()
Sexes = list()
count = 0
# =============================================================================
# for PathDicom,b,c in os.walk(FolderPath,topdown=False):
# 
#     if PathDicom != FolderPath:
#         lstFilesDCM = []  # create an empty list
# 
#         for dirName, subdirList, fileList in os.walk(PathDicom, topdown=False):
#             for filename in fileList:
#                 if ".dcm" in filename.lower():  # check whether the file's DICOM
#                     count+=1
#                     #print(os.path.join(dirName,filename))
#                     lstFilesDCM.append(os.path.join(dirName,filename))
# =============================================================================
def path():
    n = 0
    listOfPaths = list()
    for dirName, subdirList, fileList in os.walk(FolderPath, topdown=False):
        if n == 0 or n%3==0 and dirName != FolderPath:
            #print("path: ", root)
            #print("file names: ", contents)
            listOfPaths.append(dirName)
        n += 1

    return listOfPaths

listOfPaths = path()
for x in listOfPaths:
    lstFilesDCM = os.listdir(x)
    #print(lstFilesDCM)
    RefDs = dicom.read_file(x + '/' + lstFilesDCM[0])
    IDs.append(RefDs.PatientID)
    Ages.append(RefDs.PatientAge)
    Sexes.append(RefDs.PatientSex)

#print("<<<<<<<<<<<<Value of count>>>>>>>>>>>>>>", count)
#print("<<<<<<<<<<<<Lenght of lstFilesDCM>>>>>>>>>>>>>>", len(lstFilesDCM))

# Get ref file


IDs = pd.DataFrame(IDs, columns = ["Patient_ID"])
Ages = pd.DataFrame(Ages, columns = ["Patient_Age"])
Sexes = pd.DataFrame(Sexes, columns = ["Patient_Sex"])

patients = pd.concat([IDs, Sexes, Ages],axis =1, ignore_index=False)

patients.to_csv("patient_identification.csv", index = False)
