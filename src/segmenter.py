from distutils.archive_util import make_archive
import keras
import numpy as np
from PIL import Image

from src import config

class Segmenter():
	
	"""
	Load segmenter's pre-trained weights
	"""
	def __init__(self, img_size):
		self.img_size = img_size
		self.model = keras.models.load_model(config.paths['weights']['oxford'])

	def segment(self, image):
		image = image.resize((160, 160))
		masks = self.model.predict(np.expand_dims(image, axis=0))[0]
		masks = (255 * masks).astype('uint8')
		masks = np.asarray( Image.fromarray(masks).resize(self.img_size) )
		return (masks[:,:,0] + masks[:,:,2]) > masks[:,:,1]