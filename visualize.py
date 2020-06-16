import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

validation_data = pd.read_csv('validation.csv')
prediction_data = pd.read_csv('predictions.csv')

user_input = ''
image_idx = 0

d = { 0: "Cat", 1: "Dog" }

# while user_input != 'q':
#     image_filename = validation_data.loc[image_idx, "filename"]
#     img = mpimg.imread('images_validation/validation/' + image_filename)
#     imgplot = plt.imshow(img)
#     plt.text(0, -50, "Ground Truth: " + d[int(validation_data.loc[image_idx, "category"][1])])
#     plt.text(200, -50, "Prediction: " + d[prediction_data.loc[image_idx, "prediction"]])
#     plt.show()
#     user_input = input("Press Enter to Continue (q to Quit): ")
#     image_idx += 1

user_input = int(input("Enter a number: "))
for image_idx in range(user_input):
    image_filename = validation_data.loc[image_idx, "filename"]
    img = mpimg.imread('images_validation/validation/' + image_filename)
    imgplot = plt.imshow(img)
    plt.text(0, -50, "Ground Truth: " + d[int(validation_data.loc[image_idx, "category"][1])])
    plt.text(200, -50, "Prediction: " + d[prediction_data.loc[image_idx, "prediction"]])
    plt.show()
