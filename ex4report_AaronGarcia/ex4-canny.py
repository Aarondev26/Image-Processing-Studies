import cv2
import numpy as np
from matplotlib import pyplot as plt

#read the image
image = cv2.imread('test01.jpg', cv2.IMREAD_GRAYSCALE)

#check if image is loaded successfully
if image is None:
    print("Error: Could not open or find the image.")
    exit()

#apply canny edge detection
edges = cv2.Canny(image, 100, 200)

#display the original image and the edgedetected image
plt.figure(figsize=(10, 5))


#show before and after.
cv2.imshow("Original image1", image)
cv2.imshow("Filtered image1", edges)

cv2.waitKey(0)