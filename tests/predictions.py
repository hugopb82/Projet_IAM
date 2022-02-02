import argparse
import keras
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src import config
from src import utils
from src.upscalers.naive import NaiveUpscaler
from src.classifiers.routes import RoutesClassifier
from src.segmenter import Segmenter
from src.labelliser import Labelliser

def main(args):
	# model = keras.models.load_model(config.paths['best_model_gc'])
	model = keras.models.load_model('results/bestgc50.h5')

	filenames = utils.explore(config.paths['groundtruth'])
	filename = filenames[args.number]
	image = Image.open(config.paths['groundtruth'] + filename)
	arr = np.array(image)

	prediction = model.predict(np.expand_dims(arr, 0))[0]
	mask = np.argmax(prediction, axis=-1)

	plt.figure()

	plt.subplot(3, 4, 1)
	plt.imshow(arr)

	for i in range(11):
		plt.subplot(3, 4, i+2)
		if i == 0 :
			plt.title('Background')
		else :
			plt.title(config.labels[i-1])
		plt.imshow(prediction[:,:,i], vmin=0, vmax=1)

	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--number', default='0', type=int,
		help='Image number to test. Default to 0')
	args = parser.parse_args()
	main(args)