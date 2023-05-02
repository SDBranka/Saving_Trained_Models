# https://www.youtube.com/watch?v=RB-nX9jWUdM


import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# get the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # check the dataset
# print("x_train: ", x_train.shape)
# # x_train:  (60000, 28, 28)

# print("y_train: ", y_train.shape)
# # y_train:  (60000,)

# print("x_test: ", x_test.shape)
# # x_test:  (10000, 28, 28)

# print("y_test: ", y_test.shape)
# # y_test:  (10000,)


# # chart1
# #take a look at the data image
# fig = plt.figure()

# # plot 6 images
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(x_train[i], cmap="gray", interpolation="none")
#     plt.title(f"label: {y_train[i]}")
#     plt.xticks([])
# # fig.show()
# plt.show()


# # Look at a single sample
# NUM = 0
# print(f"label: {y_train[NUM]}")
# # label: 5

# print("Shape:", x_train[NUM].shape)
# # Shape: (28, 28)

# print("*"*9)
# # *********

# digit = x_train[NUM]
# print(digit)


# reshape the data -> Flatten the matrix
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000,784)

# change the data type
x_train = x_train.astype("float")
x_test = x_test.astype("float")


# normalize the data
x_train /= 255
x_test /= 255


# checking the dataset
print(x_train.shape, x_test.shape)
# (60000, 784) (10000, 784)

print("x_train max: ", max(x_train[0]))
print("x_test max: ", max(x_test[0]))

# preprocess the labels
NUM_CLASSES = 10

y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)















