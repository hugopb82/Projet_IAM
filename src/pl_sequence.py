from tensorflow import keras
import numpy as np
from PIL import Image

import config

from upscaler import Upscaler
from classifier import Classifier
from segmenter import Segmenter
from labelliser import Labelliser

class PL_Sequence(keras.utils.Sequence):
	"""Helper to iterate over the data (as Numpy arrays)."""

	def __init__(self, batch_size, img_size, filenames, method):
		self.batch_size = batch_size
		self.img_size = img_size
		self.filenames = filenames

		self.upscaler = Upscaler(method)
		self.classifier = Classifier()
		self.segmenter = Segmenter()
		self.labelliser = Labelliser(self.upscaler, self.classifier, self.segmenter)

	def __len__(self):
		return len(self.filenames) // self.batch_size

	def __getitem__(self, idx):
		"""Returns tuple (input, target) correspond to batch #idx."""
		i = idx * self.batch_size
		batch_filenames = self.filenames[i : i + self.batch_size]

		# Load input images and pseudo_labels
		x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
		y = np.zeros((self.batch_size,) + self.img_size + (10,), dtype="uint8")
		for j, filename in enumerate(batch_filenames):
			image = Image.open(config.paths['dataset'] + filename)
			x[j] = np.array(image.resize(self.img_size))
			y[j] = self.labelliser.label(image)

		return x, y