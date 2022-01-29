from PIL import ImageFilter

from src.upscalers.upscaler import Upscaler

class BlurryUpscaler(Upscaler):
	
	def __init__(self, upscale_size):
		super().__init__(upscale_size)

	"""
	Upscale an image
	"""
	def upscale(self, image):
		upscaled = image.resize(self.UPSCALE_SIZE).filter(ImageFilter.GaussianBlur(.5))
		return upscaled