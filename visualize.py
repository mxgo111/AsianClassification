import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

test_data = pd.read_csv('test.csv')
prediction_data = pd.read_csv('predictions.csv')

user_input = ''
image_idx = 0


user_input = int(input("Enter a number: "))
for image_idx in range(user_input):
    image_filename = test_data.loc[image_idx, "filename"]
    img = mpimg.imread('images_test/test/' + image_filename)
    imgplot = plt.imshow(img)

    width, height, _ = img.shape

    plt.text(0, -height/12, "Ground Truth: " + test_data.loc[image_idx, "category"])
    plt.text(width/2, -height/12, "Prediction: " + prediction_data.loc[image_idx, "prediction"])
    plt.show()

# image_idx=21
# image_filename = validation_data.loc[image_idx, "filename"]
# img = mpimg.imread('images_validation/validation/' + image_filename)
# imgplot = plt.imshow(img)
#
# width, height, _ = img.shape
#
# plt.text(0, -height/12, "Ground Truth: " + d[int(validation_data.loc[image_idx, "category"][1])])
# plt.text(width/2, -height/12, "Prediction: " + d[prediction_data.loc[image_idx, "prediction"]])
# plt.show()
