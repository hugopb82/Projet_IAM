import keras
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

labels = [
	'background',
	'chimpanzee',
	'orangutan',
	'wolf',
	'coyotee',
	'hamster',
	'guineapig',
	'cat',
	'lynx',
	'guepard',
	'jaguar'
]

model = keras.models.load_model('../results/final.h5')

filename = os.listdir('../data/dataset/')[0]

image = Image.open('../data/dataset/' + filename).resize((160,160))
arr = np.array(image)

prediction = model.predict(np.expand_dims(arr, 0))[0]
mask = np.argmax(prediction, axis=-1)
print(mask.shape)

plt.figure()

plt.subplot(3, 4, 1)
plt.imshow(arr)

for i in range(11):
	plt.subplot(3, 4, i+2)
	plt.title(labels[i])
	plt.imshow(prediction[:,:,i], vmin=0, vmax=1)

plt.show()

# plt.imshow(mask * 255 / 2, cmap='gray', vmin=0, vmax=255)
# plt.show()