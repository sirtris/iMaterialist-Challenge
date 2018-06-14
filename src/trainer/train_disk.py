################################################################################
#  This script provides a general outline of how you can train a model on GCP  #
#  Authors: Mick van Hulst, Dennis Verheijden                                  #
################################################################################

from __future__ import absolute_import
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications.xception import preprocess_input, decode_predictions, Xception
from keras.optimizers import Adam
import sklearn
import argparse
import random
import cv2
import os
#import path

import json
from tensorflow.python.lib.io import file_io
from PIL import Image
import io
from google.cloud import storage
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import keras.backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print('initializing bucket')
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/RU/Sem2/MLiP/Comp2/Team Valenteam-0eaff6cf3578.json"
#client = storage.Client()
#bucket = client.get_bucket('mlip-team-valenteam-mlengine')



"""
CONFIG
"""
BATCH_SIZE = 64
#SUBSET = None
SUBSET = None
if not SUBSET == None:
    STEPS_PER_EPOCH = SUBSET / BATCH_SIZE
else:
    STEPS_PER_EPOCH = 234843 / BATCH_SIZE
EPOCHS = 4
random.seed(42)



def prepare_input_labels(labels):
    mlb = MultiLabelBinarizer(np.arange(1,229))
    mlb.fit(labels)
    labels = mlb.transform(labels)
    return labels

def train_data_generator(data_path,batch_size):

    imagePaths = sorted(list(os.listdir(data_path)))
    random.seed(42)
    random.shuffle(imagePaths)

    L = len(imagePaths)

    while True:#this line is just to make the generator infinite, keras needs that
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            feats=[]
            labels=[]
            limit = min(batch_end,L)

            for imagePath in imagePaths[batch_start:limit]:
                file_path=os.path.join(data_path,imagePath)
                if not file_path.is_file():  #os.path.isfile(file_path):
                    continue
                img = cv2.imread(file_path)
#                print(imagePath)
#                print(type(img))
                # dubble-check if image is read correctly 
                if not img == None:     # was type(img)
                    img = cv2.resize(img, (299,299),interpolation=cv2.INTER_CUBIC)
                    img = img_to_array(img)

                    feats.append(img)
#                name = blob.name

                label=imagePath.split('_')[-1]
                label = label.replace('.jpg','')
                label = ast.literal_eval(label)
                labels.append(label)
            X = np.array(feats,dtype='float32')
            Y = prepare_input_labels(labels)
#            print(len(X))
#            print(len(labels))

            yield(X,Y)

            batch_start += batch_size
            batch_end += batch_size






def create_model(num_classes):
    """
    In here you can define your model
    NOTE: Since we are only saving the model weights, you cannot load model weights that do
    not have the exact same architecture.
    :return:
    """


    base_model = Xception(weights = 'imagenet',include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512,activation='relu')(x)
    x = Dense(512,activation='relu')(x)    ### If you want a second layer
    predictions = Dense(num_classes,activation='sigmoid')(x)

    model = Model(input=base_model.input,output=predictions)

    for layer in base_model.layers:
        layer.trainable = False


    opt = Adam()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#    model.summary()

    return model


def main():
    print('creating model')
    model = create_model(228)



    with K.tf.device('/gpu:0'):

        print('fitting model')

        model.fit_generator(train_data_generator('/home/valenteam/iMaterialist/iMaterialist-Challenge/data/',BATCH_SIZE),steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS)




    # Save model weights
    print('saving model')
    model.save('/home/valenteam/iMaterialist/iMaterialist-Challenge/src/export/Xception.h5')

    # Save model on google storage
#    with file_io.FileIO('model_InceptionV3_100000.h5', mode='r') as input_f:
#        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
#            output_f.write(input_f.read())

    print('done')

if __name__ == '__main__':
    """
    The argparser can also be extended to take --n-epochs or --batch-size arguments
    """
#    parser = argparse.ArgumentParser()
#
#    # Input Arguments
#    parser.add_argument(
#        '--job-dir',
#        help='GCS location to write checkpoints and export models',
#        required=True
#    )
#    args = parser.parse_args()
#    arguments = args.__dict__
#    print('args: {}'.format(arguments))
#
#    main(args.job_dir)
#    main('a','b','c')
#    feats,labels = load_data('As#')
    main()
