import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load input image
img1 = cv2.imread('test02.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
height, width = img1.shape
print(f'Image size: width={width}, height={height}')

# Make histogram
hist, bins = np.histogram(img1.flatten(), range=(0,256), bins=256)

# Show Histogram
plt.bar(bins[:-1], hist)
plt.show()
