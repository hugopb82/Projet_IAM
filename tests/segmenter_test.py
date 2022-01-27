import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../src/")

from segmenter import Segmenter
import config

filename = os.listdir(config.paths['dataset'])[5401]

image = Image.open(config.paths['dataset'] + filename).resize((160, 160))
arr = np.array(image)

segmenter = Segmenter()
segmented = segmenter.segment(arr)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(arr)
plt.subplot(1, 2, 2)
plt.imshow(segmented)
plt.show()