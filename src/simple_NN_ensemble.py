# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 19:42:59 2018

@author: Valentin
"""

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
from keras.optimizers import Adam, SGD

import pickle

def get_model_predictions(learning=True):
    if learning:
        folder_string = 'predictions/validation/'
        to_string = 'predictions_validation.pkl'
    else:
        folder_string = 'predictions/'
        to_string = 'predictions.pkl'
    model_names = ['DenseNet_bigset','Xception_bigModel','Xception_bigset']

    model_predictions=[]
    for i in range(len(model_names)):
        name = folder_string+model_names[i]+to_string
        with open(name,'rb') as f:
            model_predictions.append(np.vstack(pickle.load(f)))

    preds = np.array(model_predictions)
#    if learning:
    preds = np.reshape(preds,(preds.shape[1],preds.shape[0],preds.shape[2]))

    return preds

def get_true_labels():
    with open('data/labels_all_files_validation.npy','rb') as f:
        labels = np.load(f)
    zeros = np.zeros((len(labels),228))

    zeros[:,:46] = labels[:,:46]
    zeros[:,47:161] = labels[:,46:160]
    zeros[:,163:] = labels[:,160:]

    return zeros

if __name__ == '__main__':

    preds=get_model_predictions()
    X = np.reshape(preds,(preds.shape[0],-1))

    Y = get_true_labels()

    model = Sequential()
    model.add(Dense(512,activation='linear',input_shape=(X.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(512,activation='linear'))
    model.add(Dropout(0.5))
    model.add(Dense(228,activation='sigmoid'))

    opt = SGD()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    CUTOFF = 0.5
    EPOCHS = 100
    model.fit(X,Y,epochs=EPOCHS)

    X_test = get_model_predictions(False)
    X_test = np.reshape(X_test,(X_test.shape[0],-1))

    output = model.predict(X_test)

    tmp = output


    output[output>CUTOFF] = 1
    output[output<CUTOFF] = 0

    ids=np.arange(1,39707)
    str_preds = []
    for i in range(len(output)):
        curr_str_pred = ''
        for j in range(len(output[i])):
            if output[i,j]==1:
                curr_str_pred = curr_str_pred+str(j+1)+" "
        str_preds.append(curr_str_pred)


    df = pd.DataFrame({'image_id':ids,'label_id':str_preds})
    df.to_csv('predictions/NN_ensemble_v2_{}_{}.csv'.format(str(CUTOFF),str(EPOCHS)),index=False)



















