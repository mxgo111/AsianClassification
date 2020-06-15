import sys
import pandas as pd

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.python.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from data import train_generator, validation_generator

checkpoint_path = "model_checkpoints/cp-0010.ckpt"
resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
num_classes = 2

#Load model from checkpoint
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(Dense(num_classes, activation='softmax'))
model.load_weights(checkpoint_path)

predictions = model.predict(validation_generator)

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('predictions.csv')

with open('predictions.txt', 'w') as file:
    print(predictions_df, file=file)
