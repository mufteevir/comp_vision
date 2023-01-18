from keras.models import load_model
import numpy as np
from keras.preprocessing import image

new_model = load_model('car spikes left camerav3.h5')

spike_file = 'Left camera/test/2_Сhipped insert/208_L.jpeg'

spike_img = image.image_utils.load_img(spike_file, target_size=(150, 240))

spike_img = image.image_utils.img_to_array(spike_img)#spike image (150,150,3)

spike_img = np.expand_dims(spike_img, axis=0)#spike_image(1,150,150,3)
spike_img = spike_img / 255

print(np.argmax(new_model.predict(spike_img), axis=-1))
print(new_model.predict(spike_img))