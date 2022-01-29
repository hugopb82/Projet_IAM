import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../src/")

from upscaler import Upscaler
from classifier import Classifier
import config


start = 50
filenames = os.listdir(config.paths['dataset'])[start:start+9]

plt.figure()
for (i, filename) in enumerate(filenames):

	image = Image.open(config.paths['dataset'] + filename).resize((56, 56))
	arr = np.array(image)

	upscaler = Upscaler('espcn')
	upscaled = upscaler.upscale(arr)

	classifier = Classifier()
	img_class = classifier.classify(upscaled)

	plt.subplot(3, 3, i+1)
	plt.title(img_class)
	plt.imshow(upscaled)

plt.show()