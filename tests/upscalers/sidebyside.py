import argparse
from matplotlib import pyplot as plt
from PIL import Image

from src import config
from src import utils
from src.upscalers.naive import NaiveUpscaler
from src.upscalers.blurry import BlurryUpscaler
from src.upscalers.superres import SuperresUpscaler

def main(args):
	filenames = utils.explore(config.paths['groundtruth'])
	filename = filenames[args.number]
	image = Image.open(config.paths['groundtruth'] + filename)
	if args.method == "naive":
		upscaler = NaiveUpscaler((160, 160))
	elif args.method == "blurry":
		upscaler = BlurryUpscaler((160, 160))
	elif args.method == "superres":
		upscaler = SuperresUpscaler((160, 160))
	upscaled = upscaler.upscale(image)

	plt.figure(filename)
	plt.subplot(1,2,1)
	plt.title('Orginial')
	plt.imshow(image)
	plt.subplot(1,2,2)
	plt.title('Upscaled')
	plt.imshow(upscaled)
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--number', default='0', type=int,
		help='Image number to test. Default to 0')
	parser.add_argument('-m', '--method', default='naive',
		help='Upscaler method to test. Default to naive')
	args = parser.parse_args()
	main(args)