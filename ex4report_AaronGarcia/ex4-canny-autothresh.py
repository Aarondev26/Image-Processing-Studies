import cv2
import numpy as np
from matplotlib import pyplot as plt

def autocanny(image):
    #check if image is loaded successfully
    if image is None:
        print("Error: Could not open or find the image.")
        exit()

    #compute the median of the single-channel pixel intensities
    median_val = np.median(image)

    #apply automatic Canny edge detection using the computed median
    lower_threshold = int(max(0, (1.0 - 0.33) * median_val))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_val))

    #apply Canny edge detection
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

#read the images
image1 = cv2.imread('test01.jpg', cv2.IMREAD_GRAYSCALE)
edges1 = autocanny(image1)

image2 = cv2.imread('test02.jpg', cv2.IMREAD_GRAYSCALE)
edges2 = autocanny(image2)

image3 = cv2.imread('test03.jpg', cv2.IMREAD_GRAYSCALE)
edges3 = autocanny(image3)

#show before and after.
cv2.imshow("Original image1", image1)
cv2.imshow("Filtered image1", edges1)

#show before and after.
cv2.imshow("Original image2", image2)
cv2.imshow("Filtered image2", edges2)

#show before and after.
cv2.imshow("Original image3", image3)
cv2.imshow("Filtered image3", edges3)

cv2.waitKey(0)
