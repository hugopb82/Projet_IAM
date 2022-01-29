import numpy as np

class Labelliser():
	
	def __init__(self, img_size, upscaler, classifier, segmenter):
		self.img_size = img_size
		self.upscaler = upscaler
		self.classifier = classifier
		self.segmenter = segmenter

	def label(self, image):
		upscaled_image = self.upscaler.upscale(image)
		image_class = self.classifier.classify(upscaled_image)
		image_segmentation = self.segmenter.segment(upscaled_image)

		y = np.zeros(self.img_size, dtype=int)
		y[:, :] = (image_class + 1) * image_segmentation
		y = np.expand_dims(y, 2)
		
		return y