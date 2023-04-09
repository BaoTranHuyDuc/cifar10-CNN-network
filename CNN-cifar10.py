# Import libraries
import numpy as np
import tensorflow as tf
import keras

# Load the cifar10 dataset and split train/test
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Split train/valid from the training set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=5)

print("Train shape: X_train = " + str(X_train.shape) + ", y_train = " + str(y_train.shape))
print("Validation shape: X_val = " + str(X_val.shape) + ", y_val = " + str(y_val.shape))
print("Test shape: X_test = " + str(X_test.shape) + ", y_test = " + str(y_test.shape))

# Show some samples in the dataset
import matplotlib.pyplot as plt
imgplot = plt.imshow(X_train[5])
plt.show()
imgplot = plt.imshow(X_test[10])
plt.show()

X_train_norm, X_val_norm, X_test_norm = X_train/255.0, X_val/255.0, X_test/255.0

from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Conv3D, Dense, MaxPooling2D, Flatten, AveragePooling2D, ZeroPadding2D, Dropout
from keras.preprocessing.image import ImageDataGenerator

#This is the network of the architecture of my choice, accuracy is 87.68%

#I only have 2 pooling layers because the input data is small, so too many pooling layers will make the output feature map too small to make predictions on

#I slowly increase the number of filters in convolutional layers because the deeper we go, the more complex the features that conv layers will extract
#so increasing the number of filters will help with extracting more complex features

#I added drop out after max pooling to remove under-used neurons, this helps to reduce overfitting since only the useful neurons will remain

#I added batch normalization to normalized values in each neurons after each layer. This is a form of feature scaling
#This reduces the range between each value and ensure that all values are treated equally in future calculations 

#I only use padding on the first convolution layer because adding padding on all convolutional layers will increase the number of parameters by too many and cause a decrease in accuray
#Similarly, using no padding at all reduces the number of parameters by too many and cause a decrease in accuracy
#Through experimenting, I've found the sweetspot to be only applying padding the first convolution layers after initialization and pooling layer

dnn_2 = Sequential()


dnn_2.add(Conv2D(filters = 32, input_shape = (32, 32, 3), kernel_size = (3,3), activation="relu", padding = 'same'))
dnn_2.add(BatchNormalization())
dnn_2.add(Conv2D(filters = 32, kernel_size = (3,3), activation="relu"))
dnn_2.add(BatchNormalization())
dnn_2.add(Conv2D(filters = 64, kernel_size = (3,3), activation="relu"))
dnn_2.add(BatchNormalization())
dnn_2.add(MaxPooling2D(strides = 2, pool_size = (2, 2)))
dnn_2.add(Dropout(0.5))

dnn_2.add(Conv2D(filters = 64, kernel_size = (3,3), activation="relu", padding = 'same'))
dnn_2.add(BatchNormalization())
dnn_2.add(Conv2D(filters = 128, kernel_size = (3,3), activation="relu"))
dnn_2.add(BatchNormalization())
dnn_2.add(Conv2D(filters = 128, kernel_size = (3,3), activation="relu"))
dnn_2.add(BatchNormalization())
dnn_2.add(MaxPooling2D(strides = 2, pool_size = (2, 2)))
dnn_2.add(Dropout(0.5))


dnn_2.add(Flatten())
dnn_2.add(Dense(1024, activation='relu'))
dnn_2.add(BatchNormalization())
dnn_2.add(Dropout(0.5))
dnn_2.add(Dense(10, activation='softmax'))

dnn_2.summary()

dnn_2.compile(loss = tf.keras.losses.sparse_categorical_crossentropy,
                optimizer= tf.keras.optimizers.Adam(), metrics=['accuracy'])

#Running 300 epochs is unnecessary, I've found that running 100 epochs is enough
#I run 300 epochs just to maximizes my model's accuray and because GPU usage on Colab is free
history_2 = dnn_2.fit(X_train_norm, y_train, 
                    epochs=300, 
                    validation_data=(X_val_norm, y_val))

result_2 = dnn_2.evaluate(X_test_norm, y_test)
print(dnn_2.metrics_names)
print("Loss and accuracy on the test set: loss = {}, accuracy = {}".format(result_2[0],result_2[1]))