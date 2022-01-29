import argparse
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
	filenames = utils.explore(config.paths['groundtruth'])
	filename = filenames[args.number]
	image = Image.open(config.paths['groundtruth'] + filename)
	
	upscaler = NaiveUpscaler((256, 256))
	classifier = RoutesClassifier()
	segmenter = Segmenter((64, 64))

	labelliser = Labelliser((64, 64), upscaler, classifier, segmenter)
	label = labelliser.label(image)

	plt.figure()

	plt.subplot(3, 4, 1)
	plt.title('Original')
	plt.imshow(image)

	for i in range(11):
		plt.subplot(3, 4, i+2)
		if i == 0 :
			plt.title('Background')
		else :
			plt.title(config.labels[i-1])
		plt.imshow(label[:,:,0] == i, vmin=0, vmax=1)

	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--number', default='0', type=int,
		help='Image number to test. Default to 0')
	args = parser.parse_args()
	main(args)