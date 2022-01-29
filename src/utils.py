from PIL import Image
import os
import numpy as np


def load_data(path, size = (64, 64), n = -1):
	filenames = os.listdir(path)[:n]

	images = np.array(
		[np.array(Image.open(path + filename).resize(size))
		for filename in filenames]
	)

	return (filenames, images)


def explore(path):
	return os.listdir(path)