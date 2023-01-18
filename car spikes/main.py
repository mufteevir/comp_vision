"""
project for inspecting car spikes for 6 categories:
Good, No insert, Chipped insert, turned bush,
No bush, Nofounding bush, Cracking bash
"""
import warnings

from keras.layers import Activation, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

#Its usually a good idea to manipulate the images with rotation, resizing,
# and scaling so the model becomes more robust to different images that our
# data set doesn't have. We can use the ImageDataGenerator to do this automatically for us.
# Check out the documentation for a full list of all the parameters you can use here!

image_gen = ImageDataGenerator(  # rotation_range=30,  # rotate the image 30 degrees
    # width_shift_range=0.1,  # Shift the pic width by a max of 10%
    # height_shift_range=0.1,  # Shift the pic height by a max of 10%
    rescale=1 / 255,  # Rescale the image by normalzing it.
    # shear_range=0.2,  # Shear means cutting away part of the image (max 20%)
    # zoom_range=0.2,  # Zoom in by 20% max
    # horizontal_flip=True,  # Allo horizontal flipping
    # fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
)
image_gen.flow_from_directory('Left camera/train')
image_gen.flow_from_directory('Left camera/test')
# # width,height,channels
#image_shape = (1200, 1920, 3)
image_shape = (150, 240, 3)

#Training the Model
model = Sequential()
# FIRST SET OF LAYERS
# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 240, 3), activation='relu', ))
# POOLING LAYER
model.add(MaxPooling2D(pool_size=(2, 2)))
# second SET OF LAYERS
# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 240, 3), activation='relu', ))
# POOLING LAYER
model.add(MaxPooling2D(pool_size=(2, 2)))
# third SET OF LAYERS
# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(150, 240, 3), activation='relu', ))
# POOLING LAYER
model.add(MaxPooling2D(pool_size=(2, 2)))
# FLATTEN IMAGES FROM BEFORE FINAL LAYER
model.add(Flatten())
# 256 NEURONS IN DENSE HIDDEN LAYER
model.add(Dense(128))
model.add(Activation('relu'))
# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))
# LAST LAYER IS THE CLASSIFIER, THUS 7 POSSIBLE CLASSES
model.add(Dense(7, activation='softmax'))
#model.add(Dense(7, activation='sigmoid'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

batch_size = 4

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

results = model.fit_generator(train_image_gen, epochs=30,
                              steps_per_epoch=150,
                              validation_data=test_image_gen,
                              validation_steps=12)

model.save('car spikes left camerav3.h5')
