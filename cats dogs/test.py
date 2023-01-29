import numpy as np
from keras.models import load_model
from keras.preprocessing import image

new_model = load_model('cat_dog_100epochs.h5')
dog_file = 'CATS_DOGS/test/Dog/10005.jpg'

dog_img = image.image_utils.load_img(dog_file, target_size=(150, 150))

dog_img = image.image_utils.img_to_array(dog_img)

dog_img = np.expand_dims(dog_img, axis=0)
dog_img = dog_img / 255

prediction_prob = new_model.predict(dog_img)

# Output prediction
print(f'Probability that image is a dog is: {prediction_prob} ')
