from PIL import Image
import cv2
import numpy as np
import os
import tqdm
import argparse

import config

def load_data(n):
	path = config.paths['dataset']
	filenames = os.listdir(path)[:n]

	images = np.array(
		[np.array(Image.open(path + filename).resize((64, 64)))
		for filename in filenames]
	)

	return (filenames, images)

def main(args):
	sr = cv2.dnn_superres.DnnSuperResImpl_create()
	sr.readModel(config.paths['weights'][args.method])
	sr.setModel(args.method, 4)

	filenames, images = load_data(args.number)

	for i in tqdm.tqdm(range(len(images))):
		result = sr.upsample(images[i])
		result = Image.fromarray(result, "RGB")
		result.save(config.paths['dataset_upscaled'][args.method] + filenames[i])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--method', default='espcn',
        help='Upscaling method : espcn, edsr, lapsrn')
	parser.add_argument('-n', '--number', default=10, type=int,
        help='Number of images')
	args = parser.parse_args()
	main(args)