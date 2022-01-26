import cv2

import config

class Upscaler():
	
	"""
	Load the weights according to the choosen method
	"""
	def __init__(self, method='espcn'):
		self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
		self.sr.readModel(config.paths['weights'][method])
		self.sr.setModel(method, 4)

	"""
	Upscale an image
	"""
	def upscale(self, image):
		return self.sr.upsample(image)