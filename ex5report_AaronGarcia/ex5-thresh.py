import cv2
import numpy as np
from matplotlib import pyplot as plt

#read the image
image = cv2.imread('test52.jpg', cv2.IMREAD_GRAYSCALE)

#define og thresholding: compute the median of the single-channel pixel intensities
og_threshold = np.median(image)

#define delta_t to be the rate of change between current_threshold-prev_threshold
delta_t = -1

#calculate the desired global threshold. If delta_t = 0, finalize global threshold value.
prev_threshold = og_threshold
threshold = prev_threshold
while delta_t != 0:

    prev_threshold = threshold
    #debugging
    print("Threshold is now", threshold)
    width, height = image.shape
    #t1/t2 is # of all pixels in g1/g2 respectively
    t1 = 0
    t2 = 0
    #g1/g2 is a bucket of the total combined pixel value intensity. adding it makes it easier to calc m1/m2 respectively
    g1 = 0
    g2 = 0
    #algorithm used to search through image and add up all pixel intensity values
    for x in range(0, width):
        for y in range(0, height):
            if image[x,y] > threshold:
                g1 = g1 + image[x,y]
                t1 = t1 + 1
            else:
                g2 = g2 + image[x,y]
                t2 = t2 + 1
    #calculating median threshold value
    threshold = (g1/t1 + g2/t2)/2
    #calculating delta_t
    delta_t = prev_threshold - threshold 

#use static threshold
max_value = 255
_, binary_static_image = cv2.threshold(image, og_threshold, max_value, cv2.THRESH_BINARY)
print("Used threshold value:", _)

#apply calculated global threshold to the image to produce final image
_, binary_iterative_image = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
print("Used threshold value:", _)

#apply gaussian blur to reduce noise, improve output
blurred = cv2.GaussianBlur(image, (5, 5), 0)

#apply Otsu's thresholding
ret, otsu_threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#display the results
#show before and after.
cv2.imshow("Original image", image)
cv2.imshow("Global Thresholding- Static ", binary_static_image)
cv2.imshow("Global Thresholding- Iterative", binary_iterative_image)
cv2.imshow("Blurred image", blurred)
cv2.imshow("Otsu-threshold image", otsu_threshold)

cv2.waitKey(0)

