import warnings

import cv2
from keras.layers import Activation, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras.utils as image
import matplotlib.pyplot as plt

spike = cv2.imread('Left camera/train/0_Good/0_L.jpeg')
spike = cv2.cvtColor(spike, cv2.COLOR_BGR2RGB)

# print(spike.shape)
# plt.imshow(spike)
# plt.show()

image_gen = ImageDataGenerator(#rotation_range=30,  # rotate the image 30 degrees
                               #width_shift_range=0.1,  # Shift the pic width by a max of 10%
                               #height_shift_range=0.1,  # Shift the pic height by a max of 10%
                               rescale=1 / 255,  # Rescale the image by normalzing it.
                               #shear_range=0.2,  # Shear means cutting away part of the image (max 20%)
                               #zoom_range=0.2,  # Zoom in by 20% max
                               #horizontal_flip=True,  # Allo horizontal flipping
                               #fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
                               )
image_gen.flow_from_directory('Left camera/train')
image_gen.flow_from_directory('Left camera/test')
# # width,height,channels
image_shape = (1200, 1920, 3)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(1200, 1920, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(1200, 1920, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(1200, 1920, 3), activation='relu', ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
#model.add(Dense=6)
#model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
model.summary()

batch_size = 3

train_image_gen = image_gen.flow_from_directory('Left camera/train',
                                                target_size=image_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='categorical')
test_image_gen = image_gen.flow_from_directory('Left camera/test',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='categorical')
train_image_gen.class_indices

warnings.filterwarnings('ignore')

results = model.fit_generator(train_image_gen, epochs=3,
                              steps_per_epoch=150,
                              validation_data=test_image_gen,
                              validation_steps=12)




# dog_file = 'Left camera/train/0_Good/0_L.jpeg'
# dog_img = image.load_img(dog_file, target_size=(1200, 1920))
# dog_img = image.img_to_array(dog_img)
# dog_img = np.expand_dims(dog_img, axis=0)
# dog_img = dog_img / 255
#
# prediction_prob = model.predict(dog_img)
# print(f'Probability that image is a dog is: {prediction_prob} ')
