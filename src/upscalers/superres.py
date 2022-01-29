import cv2
import numpy as np
from PIL import Image

from .upscaler import Upscaler

from .. import config

class SuperresUpscaler(Upscaler):
	
	"""
	Load the weights according to the choosen method
	"""
	def __init__(self, upscale_size, method='espcn'):
		super().__init__(upscale_size)
		self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
		self.sr.readModel(config.paths['weights'][method])
		self.sr.setModel(method, 4)

	"""
	Upscale an image
	"""
	def upscale(self, image):
		upscaled = self.sr.upsample(np.asarray(image))
		upscaled = Image.fromarray(upscaled).resize(self.UPSCALE_SIZE)
		return upscaled