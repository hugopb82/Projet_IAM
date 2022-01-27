import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../src/")

from upscaler import Upscaler
from classifier import Classifier
import config

filename = os.listdir(config.paths['dataset'])[5404]

image = Image.open(config.paths['dataset'] + filename).resize((56, 56))
arr = np.array(image)

upscaler = Upscaler('espcn')
upscaled = upscaler.upscale(arr)

classifier = Classifier()
img_class = classifier.classify(upscaled)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(arr)
plt.subplot(1, 2, 2)
plt.imshow(upscaled)
plt.show()

print(img_class)