import numpy as np

class Labelliser():
	
	def __init__(self, upscaler, classifier, segmenter):
		self.upscaler = upscaler
		self.classifier = classifier
		self.segmenter = segmenter

	def label(self, image):
		upscaled_image = self.upscaler.upscale(np.array(image.resize((56,56))))
		image_class = self.classifier.classify(upscaled_image)
		image_segmentation = self.segmenter.segment(np.array(image.resize((160,160))))

		y = np.zeros((160,160))
		y[:, :] = (image_class + 1) * image_segmentation
		
		return y