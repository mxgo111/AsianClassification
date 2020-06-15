import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

validation_data = pd.read_csv('validation.csv')
prediction_data = pd.read_csv('prediction.csv')

user_input = ''
image_idx = 0

d = { "0": "Dog", "1": "Cat" }

while user_input != 'q':
    image_filename = validation_data.loc[image_idx, "filename"]
    img = mpimg.imread('images_validation/validation/' + image_filename)
    imgplot = plt.imshow(img)
    plt.text(0, 0, "Ground Truth: " + d[validation_data.loc[image_idx, "category"]])
    plt.text(0, 1, "Prediction: " + d[prediction_data.loc[image_idx, "category"]])
    plt.show()
    user_input = input("Press Enter to Continue (q to Quit): ")
    plt.close()
    image_idx += 1
