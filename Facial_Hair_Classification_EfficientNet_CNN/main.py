# Libraries
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import cv2

from tensorflow import keras
from glob import glob
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from keras import layers
from keras.callbacks import ModelCheckpoint


# Extracting Data From Zip File
with ZipFile('Facial_Hair_Dataset.zip') as zip:
    zip.extractall('Data')


# Hyperparameters
SPLIT = 0.20
BATCH_SIZE = 1
EPOCHS = 20
IMG_SIZE = 256
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


# Data Preprocessing
X = []
Y = []

data_path = 'Data/images'

classes = os.listdir(data_path)

for i, name in enumerate(classes):
    images = glob(f'{data_path}/{name}/*.jpg')
    
    for image in images:
        img = cv2.imread(image)

        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
Y = pd.get_dummies(Y)


# Data Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size= SPLIT,
                                                    random_state= 24,
                                                    shuffle= True
                                                    )


# Creating Model Based On EfficientNet
base_model = keras.applications.EfficientNetB3(input_shape= IMG_SHAPE,
                                               include_top= False,
                                               pooling= 'max'
                                               )

model = keras.Sequential([
    base_model,
    layers.Dropout(0.1),

    layers.Dropout(0.2),
    layers.Dense(128, activation= 'relu'),

    layers.Dropout(0.25),
    layers.Dense(2, activation= 'softmax')
])

model.compile(optimizer= 'adam',
              metrics= ['accuracy'],
              loss= 'categorical_crossentropy'
              )


# Creating Model Checkpoint
check = ModelCheckpoint('output/model_weights_checkpoint.h5',
                        monitor= 'val_accuracy',
                        verbose= 1,
                        save_best_only= True,
                        save_weights_only= True
                        )


# Training Model
model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          epochs= EPOCHS,
          shuffle= True,
          callbacks= check,
          verbose= 1,
          validation_data= (X_test, Y_test)
          )