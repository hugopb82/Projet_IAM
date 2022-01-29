import argparse
from random import Random
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

from src import config
from src import utils
from src.upscalers.blurry import BlurryUpscaler
from src.upscalers.superres import SuperresUpscaler
from src.classifiers.routes import RoutesClassifier

def main(args):
	filenames = utils.explore(config.paths['groundtruth'])
	Random(1337).shuffle(filenames)
	filenames = filenames[:5000]

	upscaler = BlurryUpscaler((224, 224))
	# upscaler = SuperresUpscaler((224, 224))

	classifier = RoutesClassifier()
	
	confusion_matrix = np.zeros((10, 10))
	for filename in tqdm(filenames):
		groundtruth_class = int(filename[0])
		image = Image.open(config.paths['groundtruth'] + filename)
		image_class = classifier.classify(upscaler.upscale(image))
		confusion_matrix[groundtruth_class, image_class] += 1

	print(confusion_matrix)
	plt.imshow(confusion_matrix)
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--number', default='0', type=int,
		help='Image number to test. Default to 0')
	parser.add_argument('-m', '--method', default='routes',
		help='Upscaler method to test. Default to naive')
	args = parser.parse_args()
	main(args)