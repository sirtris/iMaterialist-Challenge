# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:44:14 2018

@author: Valentin
"""

from __future__ import absolute_import
import numpy as np
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, LSTM, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
from keras.optimizers import Adam
import sklearn
import argparse
from imutils import paths
import random
import cv2
import os
import pickle
from PIL import Image

import json
from tensorflow.python.lib.io import file_io

def get_images(path):
    imagePaths = sorted(list(paths.list_images(path)))
    return imagePaths

def load_trained_model(path='D:\RU\Sem2\MLiP\Comp2\\root\\models\\model_DenseNet121.h5'):
    return load_model(path)

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def main(test_path='D:\RU\Sem2\MLiP\Comp2\\test\\test',mp='D:\RU\Sem2\MLiP\Comp2\\root\\trainer\\models\\',op='D:\RU\Sem2\MLiP\Comp2\\root\\trainer\\predictions\\'):

    models = ['Xception_bigModel.h5','model_VGG16.h5','model_Xception.h5']
    to_string = ['Xception_bigModel','VGG16_subset','Xception_subset']

    for i in range(0,1):

        model_path = mp+models[i]
        output_path = op+to_string[i]

        print('loading test images')
        imagePaths = get_images(test_path)
        print('chunking')
        chunked = chunkIt(imagePaths,5)
        print('loading model')
        model = load_trained_model(model_path)
        print('predicting')
        names=[]
        preds=[]
    #    for i in range(1,39707):
    #        with file_io.FileIO('imaterialist_challenge_data/test/'+str(i)+'.jpg', 'r') as f:
    #            img = Image.open(f)
    #            image = np.array(img)
        for chunk in chunked:
            images=[]
            print('reading images')
            for imagePath in chunk:
                image = cv2.imread(imagePath)
                image= cv2.resize(image,(299,299))
                image = img_to_array(image)

                images.append(image)
                names.append(imagePath)

            images = np.array(images)
            print('predicting')
            predictions = model.predict(images)
    #        predictions[predictions>=0.5] = 1
    #        predictions[predictions<0.5] = 0
            preds.append(predictions)
        with open(output_path+"predictions.pkl", "wb") as f:
            pickle.dump(preds, f)
        with open(output_path+"names.pkl", "wb") as f:
            pickle.dump(names, f)
    #    df = pd.DataFrame({'name':names,'preds':preds})
    #    df.to_csv(output_path+'test_submission.csv')


if __name__ == '__main__':
    main()
