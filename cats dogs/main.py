"""
The Kaggle Competition: Cats and Dogs includes 25,000 images of cats and dogs.
We will be building a classifier that works with these images and attempt to detect dogs versus cats!

The pictures are numbered 0-12499 for both cats and dogs,
thus we have 12,500 images of Dogs and 12,500 images of Cats.
"""
import warnings

from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range=30,  # rotate the image 30 degrees
                               width_shift_range=0.1,  # Shift the pic width by a max of 10%
                               height_shift_range=0.1,  # Shift the pic height by a max of 10%
                               rescale=1 / 255,  # Rescale the image by normalzing it.
                               shear_range=0.2,  # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2,  # Zoom in by 20% max
                               horizontal_flip=True,  # Allo horizontal flipping
                               fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
                               )
# Generating many manipulated images from a directory
# In order to use .flow_from_directory, you must organize the images in sub-directories.
# This is an absolute requirement, otherwise the method won't work.
# The directories should only contain images of one class, so one folder per class of images.
image_gen.flow_from_directory('CATS_DOGS/train')
image_gen.flow_from_directory('CATS_DOGS/test')
# Resizing Images
# Let's have Keras resize all the images to 150 pixels by 150 pixels once they've been manipulated.
image_shape = (150, 150, 3)
# Creating the Model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, remember its binary, 0=cat , 1=dog
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Training the Model
batch_size = 16

train_image_gen = image_gen.flow_from_directory('CATS_DOGS/train',
                                                target_size=image_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')
test_image_gen = image_gen.flow_from_directory('CATS_DOGS/test',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')
train_image_gen.class_indices

warnings.filterwarnings('ignore')
results = model.fit_generator(train_image_gen, epochs=1,
                              steps_per_epoch=150,
                              validation_data=test_image_gen,
                              validation_steps=12)
# cat_dog_100epochs
model.save('cat_dog_100epochs.h5')
