import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

path = '../../Projet/annotations/trimaps/'
filename = os.listdir(path)[2000]

image = Image.open(path + filename)
arr = np.array(image)
print(arr.dtype)

plt.imshow(arr == 3, cmap='gray')
plt.show()