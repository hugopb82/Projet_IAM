import numpy as np
from os import listdir
from PIL import Image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import resnet

class Classifier():
	
	def __init__(self):
		self.routage = {0:[366,367,369,370,372,373,374,375,377,378,381,382,384], 1:[364,365,368,371,376,379,380],
           	2:[269,270,271,268,248,250,249], 3:[277,278,279,280,272,273,274,275],
           	4:[333], 5:[338,332,331,330], 6:[383,281,282,283,284,285,286], 7:[287,291,292],
           	8:[293], 9:[290,288,289]}
		self.model = resnet.ResNet50(weights="imagenet")
		
	def classify(self, image):
		# predire
		predictions = model.predict(preprocess_input(image))

		# pourcentage de toutes les classes du routage
		percentages = {}

		# pourcentage de chaque classe de notre dataset
		# pour chaque classe de notre dataset
		for (k,l) in routage.items():
			all_percentages = []
			# pour chaque classe du routage de imagenet
			for label in l:
				# recuperer la prediction de la classe de routage de imagenet
				all_percentages.append(predictions[label])
			
			percentages[k] = max(all_percentages)

		# classe de notre dataset avec le meilleur pourcentage
		label = 0
		best_percentage = 0
		for (k,per) in percentages.items():
			if per>percentage:
				best_percentage = per
				label = k
		
		return label
