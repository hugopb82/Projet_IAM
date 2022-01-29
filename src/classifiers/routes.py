import numpy as np
from keras.applications import resnet
from keras.applications.imagenet_utils import preprocess_input

from .. import config
from .classifier import Classifier

class RoutesClassifier(Classifier):
	
	def __init__(self, *args):
		super().__init__(*args)
		self.ROUTES = [[] for i in range(10)]
		self.ROUTES[config.labels.index('cat')] = [383,281,282,283,284,285,286]
		self.ROUTES[config.labels.index('lynx')] = [287,291,292]
		self.ROUTES[config.labels.index('wolf')] = [269,270,271,268,248,250,249]
		self.ROUTES[config.labels.index('coyote')] = [277,278,279,280,272,273,274,275]
		self.ROUTES[config.labels.index('cheetah')] = [293]
		self.ROUTES[config.labels.index('jaguar')] = [290,288,289]
		self.ROUTES[config.labels.index('chimpanzee')] = [366,367,369,370,372,373,374,375,377,378,381,382,384]
		self.ROUTES[config.labels.index('orangutan')] = [364,365,368,371,376,379,380]
		self.ROUTES[config.labels.index('hamster')] = [333]
		self.ROUTES[config.labels.index('guinea_pig')] = [338,332,331,330]

		self.model = resnet.ResNet50(weights="imagenet")
		
	def classify(self, image):
		image = image.resize((224, 224))
		predictions = self.model.predict(preprocess_input(np.expand_dims(image, axis=0)))[0]
		percentages = [max(predictions[self.ROUTES[i]]) for i in range(10)]
		return percentages.index(max(percentages))