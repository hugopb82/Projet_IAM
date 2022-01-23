import keras
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import config

# model = keras.models.load_model('model.h5')
model = keras.models.load_model(config.paths['weights']['oxford'])

fn = 'img10001.jpg'

image = np.array(
	Image.open(config.paths['dataset'] + fn).resize((160,160))
)

image_up = np.array(
	Image.open(config.paths['dataset_upscaled']['espcn'] + fn).resize((160,160))
)

p = model.predict(np.expand_dims(image, axis=0))[0]
p = np.argmax(p, axis=-1)
p = np.expand_dims(p, axis=-1)
p = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(p))

p_up = model.predict(np.expand_dims(image_up, axis=0))[0]
test = p_up
p_up = np.argmax(p_up, axis=-1)
p_up = np.expand_dims(p_up, axis=-1)
p_up = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(p_up))

plt.figure()
plt.subplot(2,2,1)
plt.imshow(image)

plt.subplot(2,2,2)
plt.imshow(p)

plt.subplot(2,2,3)
plt.imshow(image_up)

plt.subplot(2,2,4)
# plt.imshow(np.maximum(test[:,:,0], test[:,:,2]))
plt.imshow((test[:,:,0] + test[:,:,2]) / 2)
plt.show()