import pandas as pd
import os

filenames = os.listdir("./images_train/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append('\'1\'')
    elif category == 'cat':
        categories.append('\'0\'')
    else:
        continue

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

df.to_csv('train.csv')

filenames = os.listdir("./images_validation/validation")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append('\'1\'')
    elif category == 'cat':
        categories.append('\'0\'')
    else:
        continue

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

df.to_csv('validation.csv')
