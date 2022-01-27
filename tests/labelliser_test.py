import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../src/")

from upscaler import Upscaler
from classifier import Classifier
from segmenter import Segmenter
from labelliser import Labelliser
import config

filename = os.listdir(config.paths['dataset'])[5404]

image = Image.open(config.paths['dataset'] + filename)
arr = np.array(image)

upscaler = Upscaler('espcn')
classifier = Classifier()
segmenter = Segmenter()

labelliser = Labelliser(upscaler, classifier, segmenter)

label = labelliser.label(image)


plt.figure()

plt.subplot(3, 4, 1)
plt.imshow(arr)

for i in range(10):
	plt.subplot(3, 4, i+2)
	plt.imshow(label[:,:,i])

plt.show()