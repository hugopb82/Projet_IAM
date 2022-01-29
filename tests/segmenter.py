import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src import config
from src import utils
from src.segmenter import Segmenter
from src.upscalers.naive import NaiveUpscaler

def main(args):
	filenames = utils.explore(config.paths['groundtruth'])
	filename = filenames[args.number]
	image = Image.open(config.paths['groundtruth'] + filename)
	
	upscaler = NaiveUpscaler((160, 160))
	upscaled = upscaler.upscale(image)

	segmenter = Segmenter((64, 64))
	segmented = segmenter.segment(image)

	plt.figure(filename)

	plt.subplot(1, 2, 1)
	plt.title('Original')
	plt.imshow(image)
	plt.subplot(1, 2, 2)
	plt.title('Mask')
	plt.imshow(segmented)

	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--number', default='0', type=int,
		help='Image number to test. Default to 0')
	args = parser.parse_args()
	main(args)