#Reference: https://github.com/smopart/lung_nodule/tree/master/project/code
#Modified by: Biswas, T.

import numpy as np
import pandas as pd
import os
import dicom
import pickle
import random
from extract_array import extract_array
import pdb

#Load data
final_train = pd.read_csv("train_data.csv")
#print(final_train)
patient_id = pd.read_csv("patient_identification.csv")
#print(patient_id)
test_data = pd.read_csv("test_data.csv")

#Load pickled arrays
#train data
#first_ten = pickle.load(open("pickledata/first_ten_arrays.p", "rb" ))
#print(first_ten)

#test/val data
#second_ten = pickle.load(open("pickledata/second_ten_arrays.p", "rb" ))
#third_ten = pickle.load(open("pickledata/third_ten_arrays.p", "rb" ))
#fourth_ten = pickle.load(open("pickledata/fourth_ten_arrays.p", "rb" ))
#fifth_ten = pickle.load(open("pickledata/fifth_ten_arrays.p", "rb" ))
#sixth_ten = pickle.load(open("pickledata/sixth_ten_arrays.p", "rb" ))
seventh_ten = pickle.load(open("pickledata/seventh_ten_arrays.p", "rb" ))
#pdb.set_trace()

#training data
patient = patient_id.iloc[0:10]
#extract_array(first_ten, patient, final_train, 300, 32, 32, 11)

#test/val data
#extract_array(second_ten, patient, test_data, 10, 32, 32, 9)
#extract_array(third_ten, patient, test_data, 10, 32, 32, 9)
#extract_array(fourth_ten, patient, test_data, 10, 32, 32, 9)
#extract_array(fifth_ten, patient, test_data, 10, 32, 32, 9)
#extract_array(sixth_ten, patient, test_data, 10, 32, 32, 9)
extract_array(seventh_ten, patient, test_data, 10, 32, 32, 9)