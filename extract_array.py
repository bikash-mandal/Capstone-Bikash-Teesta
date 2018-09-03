#Reference: https://github.com/smopart/lung_nodule/tree/master/project/code
#Modified by: Biswas, T.

import numpy as np
import pandas as pd
import os
import dicom
import pickle
import random
import pdb

def extract_array(arraylist, patient, train_test_df, sample_num, xdim, ydim, zdim):
    
    positive_case_list =list()
    positive_case_names = list()
    positive_x = list()
    positive_y = list()
    positive_z = list()


    negative_case_list = list()
    negative_case_names = list()
    negative_x = list()
    negative_y = list()
    negative_z = list()

    arraynum = len(arraylist)
    
    casesnum_pos = sample_num/arraynum
    casesnum_neg = sample_num/arraynum

    ## Identify number of nodules per Scan Number
    '''nodulenum_list = list()
    for i in range(len(patient)):
        nodulenum_list.append(len(test[test["Scan Number"] == patient["Patient_ID"].iloc[i]]))'''

    for i in range(len(train_test_df)):
        train_test_df["Scan Number"][i] = train_test_df["Scan Number"][i].lower()

    #pdb.set_trace()
    xcut = (xdim -1)/2
    ycut = (ydim - 1)/2
    zcut = (zdim -1)/2


    for num in range(arraynum):
        case_counter_pos = 0
        case_counter_neg = 0

#       nodulenum = nodulenum_list[num]
        xaxis = arraylist[num].shape[0]
        yaxis = arraylist[num].shape[1]
        zaxis = arraylist[num].shape[2]

      
        xcoord = float(train_test_df[train_test_df["Scan Number"] == patient["Patient_ID"].iloc[num].lower()]["nodule_x"])
        ycoord = float(train_test_df[train_test_df["Scan Number"] == patient["Patient_ID"].iloc[num].lower()]["nodule_y"])
        zcoord = float(train_test_df[train_test_df["Scan Number"] == patient["Patient_ID"].iloc[num].lower()]["nodule_z"])

        xmin = xcoord - xcut
        xmax = xcoord + xcut
        ymin = ycoord - ycut
        ymax = ycoord + ycut
        zmin = zcoord - zcut
        zmax = zcoord + zcut
        
        diag = int(train_test_df[train_test_df["Scan Number"] == patient["Patient_ID"].iloc[num].lower()]["diag_bin"])
        if diag == 1:
            while case_counter_pos < int(casesnum_pos):
                xrand = round(random.uniform(xmin,xmax),1)
                yrand = round(random.uniform(ymin,ymax),1)
                xlow = int(xrand - xcut)
                xhigh = int(xrand + xcut + 1)
                ylow = int(yrand - ycut)
                yhigh = int(yrand + ycut + 1)

                positive_case_list = arraylist[num][xlow:xhigh, ylow:yhigh, int(zmin):int(zmax + 1)].tolist()
                positive_array = np.array(positive_case_list)
                #pdb.set_trace()
                
                if 'Training' in patient["Patient_ID"].iloc[num]: 
                    pickle.dump(positive_array, open("data/train/1/{}-{}.p".format(patient["Patient_ID"].iloc[num], case_counter_pos), "wb" ))
                else:
                    pickle.dump(positive_array, open("data/val/1/{}-{}.p".format(patient["Patient_ID"].iloc[num], case_counter_pos), "wb" ))
                
                #positive_case_list.append(arraylist[num][xlow:xhigh, ylow:yhigh, int(zmin):int(zmax + 1)].tolist())
                #positive_x.append()
                #pdb.set_trace()
                case_counter_pos += 1
                
        else:
            while case_counter_neg < int(casesnum_neg):
                xrand = xrand = round(random.uniform(xmin,xmax),1)
                yrand = round(random.uniform(ymin,ymax),1)
                xlow = int(xrand - xcut)
                xhigh = int(xrand + xcut + 1)
                ylow = int(yrand - ycut)
                yhigh = int(yrand + ycut + 1)

                #alternative: to traverse randomly through x-y coordinate except the nodule coordinate
                '''xhl = random.randint(0,1)
                yhl = random.randint(0,1)
                
                if xhl == 0:
                    xrand = random.randint(0,xmin-xcut)
                elif xhl == 1:
                    xrand = random.randint(xmax+xcut, xaxis)

                if yhl == 0:
                    yrand = random.randint(0,ymin-ycut)
                elif yhl == 1:
                    yrand = random.randint(ymax+ycut, yaxis)'''
                #pdb.set_trace()
                
                negative_case_list = arraylist[num][xlow:xhigh, ylow:yhigh, int(zmin):int(zmax + 1)].tolist()
                negative_array = np.array(negative_case_list)

                if 'Training' in patient["Patient_ID"].iloc[num]: 
                    pickle.dump(negative_array, open("data/train/0/{}-{}.p".format(patient["Patient_ID"].iloc[num], case_counter_neg), "wb" ))
                else:
                    pickle.dump(negative_array, open("data/val/0/{}-{}.p".format(patient["Patient_ID"].iloc[num], case_counter_neg), "wb" ))
                #pdb.set_trace()
                
                case_counter_neg += 1
    #return positive_case_list, negative_case_list
