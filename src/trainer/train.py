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
#from imutils import paths
import random
import cv2
import os

import json
from tensorflow.python.lib.io import file_io


def load_data(path):
    """
    Loading the data and turning it into a pandas dataframe
    :param path: Path to datafile; Can be predefined as shown above.
    :return: pandas dataframe
    """
    # imagePaths = sorted(list(paths.list_images(args["dataset"])))
    # random.seed(42)
    # random.shuffle(imagePaths)
    #

#    for imagePath in imagePaths:
#         image = cv2.imread(imagePath)
#         image = img_to_array(image)
#
#         feats.append(image)


#    file_path = 'D:\RU\Sem2\MLiP\Comp2\Fashion_MLiP\data'
#    df = pd.read_pickle(file_path+'\Training_subset.pkl')
#    IMAGE_DIMS = (299,299, 3)
##    feats = np.zeros((len(df),299,299,3))
#    feats = []
#    labels = []
#
#    for index,row in df.iterrows():
#        if not os.path.isfile(file_path+'\\'+row['FileName']+'.jpg'):
#            continue
#        else:
#
#            image = cv2.imread(file_path+'\\'+row['FileName']+'.jpg')
#            image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
#            image = img_to_array(image)
##            feats[index]=image
#            feats.append(image)
#
#            labels.append(row['Labels'])
#
##        if index == 100:
##            break
#
##    feats = preprocess_input(feats)
##    data = pd.DataFrame({'data':feats,'labels':labels})
##    data = pd.read_pickle('data_df.pkl')
#    feats = np.array(feats)
#    mlb = sklearn.preprocessing.MultiLabelBinarizer()
#    labels = mlb.fit_transform(labels)
    with file_io.FileIO(path+'/data/feats.npy', 'r') as f:
        feats=np.load(f)
    with file_io.FileIO(path+'/data/labels.npy', 'r') as f:
        labels=np.load(f)
    return feats,labels


def train_test_split_pandas(df, test_split=.2):
    """
    Naive train test split function for pandas dataframes.
    :param df:
    :param test_split:
    :return:
    """
    X = np.asarray(df['data'])
    y = np.asarray(df['labels'])

#    mlb = sklearn.preprocessing.MultiLabelBinarizer()
#    y = mlb.fit_transform(y)

    return train_test_split(X, y, test_size=test_split)


def create_model(num_classes):
    """
    In here you can define your model
    NOTE: Since we are only saving the model weights, you cannot load model weights that do
    not have the exact same architecture.
    :return:
    """
    # model = Sequential()
    # model.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(3 , 100, 100)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(32, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Convolution2D(64,(3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    base_model = Xception(weights = 'imagenet',include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512,activation='relu')(x)
    predictions = Dense(num_classes,activation='sigmoid')(x)

    model = Model(input=base_model.input,output=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(228))
    # model.add(Activation('sigmoid'))

    opt = Adam()
#    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def main(train_file, job_dir): # test_file   as second arg
    print('loading model')
    feats,labels = load_data(train_file)
    print('splitting the data')
    X_train, X_validation, y_train, y_validation = train_test_split(feats,labels,test_size=0.2)
    print('creating model')
    model = create_model(len(labels[0]))
    print('augmenting images')
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode="nearest")
    print('fitting model')
#    H = model.fit_generator(aug.flow(X_train, y_train, batch_size=500), validation_data=(X_validation,y_validation),
#            steps_per_epoch=len(X_train)/500,epochs=5,verbose=1)
    model.fit(X_train, y_train, nb_epoch=20, batch_size=50, verbose=1)

    score, accuracy = model.evaluate(X_validation, y_validation)
    print('Test score:', score)
    print('Test accuracy:', accuracy)

#    X_test = load_data(test_file)
#
#    predictions = model.predict(X_test)
#    predictions[predictions>=0.5] = 1
#    predictions[predictions<0.5] = 0
    # TODO: Kaggle competitions accept different submission formats, so saving the predictions is up to you

    # Save model weights
    model.save('model_Xception.h5')

    # Save model on google storage
    with file_io.FileIO('model_Xception.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model_Xception.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    """
    The argparser can also be extended to take --n-epochs or --batch-size arguments
    """
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )

    #parser.add_argument(
    #  '--test-file',
    #  help='GCS or local paths to test data',
    #  required=True
    #)

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    print('args: {}'.format(arguments))



    #main(args.train_file, args.test_file, args.job_dir)
    main(**arguments)
#    main('a','b','c')
#    feats,labels = load_data('As#')