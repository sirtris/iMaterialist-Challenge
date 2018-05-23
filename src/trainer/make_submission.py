# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:23:36 2018

@author: Valentin
"""

import pandas as pd
import numpy as np
import pickle

ids = np.arange(1,39707)
with open("full_predictions.pkl", "rb") as f:
        preds= pickle.load(f)


str_preds = []
for i in range(len(preds)):
    curr_str_pred = ''
    for j in range(len(preds[i])):
        if preds[i,j]==1:
            curr_str_pred = curr_str_pred+str(j)+" "
    str_preds.append(curr_str_pred)


df = pd.DataFrame({'image_id':ids,'label_id':str_preds})
df.to_csv('inception_v3_subset_submission.csv',index=False)