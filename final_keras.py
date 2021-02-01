#./bin/spark-submit

## Authors: Claudia Sanchis, Sara Pérez e Iker Atienza.
## Description: In this script we use Sparkl dataframes and keras for the correct
## prediction of the Pneumonia X-Ray dataset. We build our neuronal network using
## a keras model. 


## Initializing Spark Session and Spark Context
import pyspark
import findspark
from pyspark.sql import SparkSession

findspark.init()
spark = SparkSession.builder.appName("DL with Spark Deep Cognition").getOrCreate()
sc = spark.sparkContext


## Import packages and modules needed.
import keras
import tensorflow
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential #for neural network models
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from keras.utils import to_categorical #For One-hot Encoding
from keras.optimizers import Adam, SGD, RMSprop #For Optimizing the Neural Network
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt #Ploting charts
from glob import glob #retriving an array of files in directories
#from sparkdl.udf.keras_image_model import registerKerasImageUDF
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from pyspark.ml.pipeline import PipelineModel
from pyspark import SparkFiles
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json
import json

#Set our directory files
test_normal = "chest_xray/test/NORMAL/"
test_pneumonia = 'chest_xray/test/PNEUMONIA/'
train_normal = "chest_xray/train/NORMAL/"
train_pneumonia = 'chest_xray/train/PNEUMONIA/'
val_normal = "chest_xray/val/NORMAL/"
val_pneumonia = 'chest_xray/val/PNEUMONIA/'

# Loading the images as spark DataFrames
normal_test = spark.read.format("image").load(test_normal).withColumn("label", lit(0))
pneumonia_test = spark.read.format("image").load(test_pneumonia).withColumn("label", lit(1))
normal_train = spark.read.format("image").load(train_normal).withColumn("label", lit(0))
pneumonia_train = spark.read.format("image").load(train_pneumonia).withColumn("label", lit(1))
normal_val = spark.read.format("image").load(val_normal).withColumn("label", lit(0))
pneumonia_val = spark.read.format("image").load(val_pneumonia).withColumn("label", lit(1))

# Dataframe for training a classification model
train = normal_train.unionAll(pneumonia_train)
# Dataframe for testing the classification model
test = normal_test.unionAll(pneumonia_test)
# Dataframe for validating the classification model
val = normal_val.unionAll(pneumonia_val)

# Each of the partitions is fully loaded in memory, this might be very expensive. 
# Ensure that each of the partitions has a small size. 
train = train.repartition(100)
test = test.repartition(100)
val = val.repartition(100)

# Set our model fit parameters
epochs = 20
batch_size = 16
verbose = 2
validation_steps = 10

# Reduce the image scale for homogeneus images shape
img_width, img_height = 150, 150 #image dimensions
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Build the different layers of our neural network
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Model compilation using an optimization process
optimizer = Adam(lr = 0.0001)
early_stopping_monitor = EarlyStopping(patience = 3, monitor = "val_acc", mode="max", verbose = 2)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)

# Model fit
modelfit = model.fit(train,epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=val, validation_steps=validation_steps)

# Model save. 
with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model_weights.h5')

# Use our model to predict the pneumonia chest x-ray
score = model.evaluate(test, verbose=2, steps=100)

print("Test set accuracy = " + score[1])