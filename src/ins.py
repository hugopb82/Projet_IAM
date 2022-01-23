from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.applications import resnet
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import os
from PIL import Image

import config

import json
class_idx = json.load(open(config.paths['imagenet_labels']))
idx2label = np.array([class_idx[str(k)][1] for k in range(len(class_idx))])

model = resnet.ResNet50(weights='imagenet')
ins = np.zeros((10, 1000))

def compute_ins(images, i):
	predictions = model.predict(preprocess_input(images))
	sum = np.sum(predictions, axis=0)
	ins[i, :] += sum
	ins[i, :] /= np.sum(ins[i, :])
	

labels = [
	'cat',
	'lynx',
	'jaguar',
	'cheetah',
	'wolf',
	'coyote',
	'chimpanzee',
	'orangutan',
	'hamster',
	# 'guinea_pig',
]
labels = [
	'jaguar',
	'cheetah'
]

# Load files
for (i, label) in enumerate(labels):
	path = config.paths['datasignature'] + label + '/'
	filenames = os.listdir(path)[:2]

	images = np.array(
		[np.array(Image.open(path + filename).resize((224, 224)))
		for filename in filenames]
	)

	compute_ins(images, i)

# Show results
n = 10
for (i, label) in enumerate(labels):
	idx = np.argsort(ins[i, :])[::-1][:n]
	plt.figure(label)
	plt.bar(idx2label[idx], ins[i, idx])

plt.show()