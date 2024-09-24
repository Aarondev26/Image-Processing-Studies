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

#make copy to prepare normalization
hist_norm = hist.astype(float)

#normalize
for i, j in np.ndenumerate(hist):
    hist_norm[i] /= 250000

cumsum_hist = np.cumsum(hist_norm)

#for each pixel in the image, find intensity value, and set the correspon
equalized_image = cumsum_hist[img1]

# Calculate histogram of equalized image
# Set "range=(0,1)" since the equalized histogram uses that range for its intensity values.
equalized_hist, _ = np.histogram(equalized_image.flatten(), range=(0,1), bins=256)

# Show original and equalized image
cv2.imshow("Original", img1)
cv2.imshow("Equalized", equalized_image)

# Show original image and its histogram
plt.subplot(2, 1, 1)
plt.bar(bins[:-1], hist, width=1)
plt.title('Original Histogram')

# Show equalized image and its histogram
plt.subplot(2, 1, 2)
plt.bar(bins[:-1], equalized_hist, width=1)
plt.title('Equalized Histogram')

# Show the layout
plt.tight_layout()
plt.show()
cv2.waitKey(0)