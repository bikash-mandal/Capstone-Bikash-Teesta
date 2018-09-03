#Reference: https://github.com/smopart/lung_nodule/tree/master/project/code
#Modified by: Biswas, T. & Mandal, B.

import numpy as np
import pandas as pd
import pickle
import dicom
import os
import pdb

ArrayList = list()

FolderPath = "/home/asif/Desktop/CapstoneData/data7/"

'''for PathDicom,b,c in os.walk(FolderPath):

    if PathDicom != FolderPath:
        lstFilesDCM = []  # create an empty list

        for dirName, subdirList, fileList in os.walk(PathDicom):
            for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName,filename))'''

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
for p in listOfPaths:
    lstFilesDCM = os.listdir(p)
    #print(lstFilesDCM)
    #pdb.set_trace()

    # Get ref file
    RefDs = dicom.read_file(p + '/' + lstFilesDCM[0])

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    #pdb.set_trace()

    locations = list()

    for i in range(len(lstFilesDCM)):
        #pdb.set_trace()
        location = dicom.read_file(p + '/' + lstFilesDCM[i])
        locations.append(int(location.SliceLocation))

    locations = np.array(locations)
    number_slices = locations.max()-locations.min()
    minimum = locations.min()

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(p + '/' + filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, (int(ds.SliceLocation)-int(minimum))] = ds.pixel_array
        #pdb.set_trace()
    ArrayList.append(ArrayDicom)
#pdb.set_trace()

#first_ten = ArrayList[0:10]
#second_ten = ArrayList[0:10]
#third_ten = ArrayList[0:10]
#fourth_ten = ArrayList[0:10]
#fifth_ten = ArrayList[0:10]
#sixth_ten = ArrayList[0:10]
seventh_ten = ArrayList[0:10]

#dir = os.path.dirname(os.path.realpath(__file__))

#np.save("train.npy",first_ten)

#pickle.dump(first_ten, open( "pickledata/first_ten_arrays.p", "wb" ))
#pickle.dump(second_ten, open( "pickledata/second_ten_arrays.p", "wb" ))
#pickle.dump(third_ten, open( "pickledata/third_ten_arrays.p", "wb" ))
#pickle.dump(fourth_ten, open( "pickledata/fourth_ten_arrays.p", "wb" ))
#pickle.dump(fifth_ten, open( "pickledata/fifth_ten_arrays.p", "wb" ))
#pickle.dump(sixth_ten, open( "pickledata/sixth_ten_arrays.p", "wb" ))
pickle.dump(seventh_ten, open( "pickledata/seventh_ten_arrays.p", "wb" ))
