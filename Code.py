import tensorflow as tf
tf.test.is_gpu_available()
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_imgl
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D
IMAGE_SIZE = [224, 224]

train_path = 'D:/XR_MERGE'
valid_path = 'D:/XR_MERGE_valid'
mobile = MobileNet(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in mobile.layers:
    layer.trainable = False
    flattened = tf.keras.layers.Flatten()(mobile.output)
fc1 = tf.keras.layers.Dense(1024, activation='relu', name="AddedDense1")(flattened)
d = tf.keras.layers.Dropout(0.25)(fc1)
fc2 = tf.keras.layers.Dense(512, activation='relu', name="AddedDense2")(d)
d2= tf.keras.layers.Dropout(0.25)(fc2)
fc3 = tf.keras.layers.Dense(256, activation='relu', name="AddedDense3")(d2)
d3=tf.keras.layers.Dropout(0.25)(fc2)
fc4 = tf.keras.layers.Dense(14, activation='softmax', name="AddedDense4")(d3)
model = tf.keras.models.Model(inputs=mobile.input, outputs=fc4)
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   preprocessing_function=preprocess_input,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('D:/XR_MERGE',
                                                 target_size = (224, 224),
                                                 batch_size = 40,
                                                 class_mode = 'categorical',
                                               
                                                 shuffle=True)
test_set = test_datagen.flow_from_directory('D:/XR_MERGE_valid',
                                            target_size = (224, 224),
                                            batch_size = 40,
                                            class_mode = 'categorical',
                                            shuffle=True)
# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=80,
 # verbose=1,
 # steps_per_epoch=len(training_set),
  #validation_steps=len(test_set)
  validation_steps = int(len(test_set)/64)

)    


