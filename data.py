import pandas as pd

from tensorflow.python.keras.applications.resnet import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

df_train = pd.read_csv('train.csv')
df_validation = pd.read_csv('validation.csv')

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_dataframe(
    df_train,
    "images_train/train",
    x_col='filename',
    y_col='category',
    target_size=(image_size, image_size),
    batch_size=24,
    class_mode='categorical')

validation_generator = data_generator.flow_from_dataframe(
    df_validation,
    "images_validation/validation",
    x_col='filename',
    y_col='category',
    target_size=(image_size, image_size),
    batch_size=24,
    class_mode='categorical')
