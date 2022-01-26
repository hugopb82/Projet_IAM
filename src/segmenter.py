import keras
import numpy as np

import config

class Segmenter():
	
	"""
	Load segmenter's pre-trained weights
	"""
	def __init__(self, ):
		self.model = keras.models.load_model(config.paths['weights']['oxford'])

	def segment(self, image):
		masks = self.model.predict(np.expand_dims(image, axis=0))[0]
		return (masks[:,:,0] + masks[:,:,2]) / 2