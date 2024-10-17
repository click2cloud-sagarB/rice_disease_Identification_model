import numpy as np
import pandas as pd
import tensorflow as tf
import json
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D , Flatten ,Dense ,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, TopKCategoricalAccuracy, sparse_top_k_categorical_accuracy
import os
from azure.storage.blob import BlobServiceClient

img_size= (224,224)
input_shape_3D = (224,224,3)
seed = 1
batch_size = 64
epochs = 30

# Load Original Image data
directory = "/home/click2cloud/originaldata/Original_dataset"

train_data = tf.keras.utils.image_dataset_from_directory(directory=directory,
                                                  labels='inferred',
                                                  label_mode='categorical',
                                                  class_names=None ,
                                                  color_mode='rgb',
                                                  image_size=img_size,
                                                  seed=seed,                                                                
                                                  validation_split=0.3,
                                                  subset='training',
                                                  )

val_data = tf.keras.utils.image_dataset_from_directory(directory=directory,
                                                  labels='inferred',
                                                  label_mode='categorical',
                                                  class_names=None ,
                                                  color_mode='rgb',
                                                  image_size=img_size,
                                                  seed=seed,
                                                  validation_split=0.3,
                                                  subset='validation',
                                                  )

# Extract the class names from the dataset
class_names = train_data.class_names

# Apply rescaling using the Rescaling layer
train_data=train_data.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))
val_data=val_data.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))

# Define Model
model = Sequential()
model.add(Conv2D(filters=16 ,kernel_size=3 ,padding='same' ,strides=1 ,activation='relu' ,use_bias=False ,input_shape=input_shape_3D))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=32 ,kernel_size=3 ,padding='same' ,strides=1 ,activation='relu' ,use_bias=False))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=64 ,kernel_size=3 ,padding='same' ,strides=1 ,activation='relu' ,use_bias=False))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=128 , kernel_size=3 ,padding='same' ,strides=1 ,activation='relu' ,use_bias=False))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=256 , kernel_size=3 ,padding='same' ,strides=1 ,activation='relu' ,use_bias=False))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(250,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

# model.summary()

Define the model with optimizer , loss , metrics
model.compile(optimizer='Adam',loss = 'categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), TopKCategoricalAccuracy(k=3)])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_data,
                    epochs=3,
                    validation_data=val_data,
                    callbacks=[early_stopping],
                    verbose=1,
                    shuffle=True)

# print Accuracy
test_Accuracy = model.evaluate(val_data)
print(f"Model's Accuracy : {test_Accuracy[1]*100}")

best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1
print(f"Best weights were obtained at epoch {best_epoch}")
# # best_weights = model.get_weights()
# # print("Best Weights:", best_weights)

class_indices = {index: name for index, name in enumerate(class_names)}

with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
    
model.save("rice_model.h5")