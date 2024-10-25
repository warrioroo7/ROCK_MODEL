# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure eager execution is enabled
tf.config.run_functions_eagerly(True)  # Only if necessary

# Model creation
model = Sequential()

# Convolutional layer with 32 filters, (3,3) kernel size, and input shape of 64x64x3
model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Max pooling layer with pool size of (2,2)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer to convert 2D output to 1D for dense layers
model.add(Flatten())

# Fully connected layers with 'relu' activation
model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))

# Output layer with 13 units (for 13 categories) and 'softmax' activation
model.add(Dense(units=13, kernel_initializer="uniform", activation="softmax"))

# Image data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Change the paths to your dataset
x_train = train_datagen.flow_from_directory(
    r'C:\Users\Plus One\Desktop\Codeutsav\llSPS-INT-3797-Rock-identification-using-deep-convolution-neural-network\dataset\dataset\trainset',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

x_test = test_datagen.flow_from_directory(
    r'C:\Users\Plus One\Desktop\Codeutsav\llSPS-INT-3797-Rock-identification-using-deep-convolution-neural-network\dataset\dataset\testset',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Checking class indices
print(x_train.class_indices)

# Compiling the model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

# Training the model
model.fit(x_train, steps_per_epoch=len(x_train), epochs=25, validation_data=x_test, validation_steps=len(x_test))

# Saving the trained model
model.save("trained_model.h5")


