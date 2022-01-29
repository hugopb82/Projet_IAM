from src.upscalers.upscaler import Upscaler

class NaiveUpscaler(Upscaler):
	
	def __init__(self, upscale_size):
		super().__init__(upscale_size)

	"""
	Upscale an image
	"""
	def upscale(self, image):
		upscaled = image.resize(self.UPSCALE_SIZE)
		return upscaled