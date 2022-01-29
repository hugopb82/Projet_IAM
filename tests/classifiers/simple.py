import argparse
from matplotlib import pyplot as plt
from PIL import Image

from src import config
from src import utils
from src.upscalers.blurry import BlurryUpscaler
from src.upscalers.superres import SuperresUpscaler
from src.classifiers.routes import RoutesClassifier

def main(args):
	filenames = utils.explore(config.paths['groundtruth'])
	filename = filenames[args.number]
	image = Image.open(config.paths['groundtruth'] + filename)

	# upscaler = BlurryUpscaler((224, 224))
	upscaler = SuperresUpscaler((224, 224))
	upscaled = upscaler.upscale(image)

	if args.method == "routes":
		classifier = RoutesClassifier()
	image_class = classifier.classify(upscaled)

	plt.figure(filename)
	# plt.title(config.labels[image_class])
	plt.title(image_class)
	plt.imshow(upscaled)
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--number', default='0', type=int,
		help='Image number to test. Default to 0')
	parser.add_argument('-m', '--method', default='routes',
		help='Upscaler method to test. Default to naive')
	args = parser.parse_args()
	main(args)