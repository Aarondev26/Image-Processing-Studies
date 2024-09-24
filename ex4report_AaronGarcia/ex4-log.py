import cv2
import numpy as np
from matplotlib import pyplot as plt

#read the input image
image = cv2.imread('test01.jpg')

#check if the image was successfully loaded
if image is None:
    print("Error: Could not open or find the image.")
else:
    #apply GaussianBlur
    #the parameters are: (source_image, (kernel_width, kernel_height), sigmaX)
    blurred_image = cv2.GaussianBlur(image, (15, 15), 1)

    #apply Laplacian Image 
    laplacian_image = cv2.Laplacian(image, cv2.CV_64F)

    laplacian_image = cv2.convertScaleAbs(laplacian_image)

    
#show before and after.
cv2.imshow("Original", image)
cv2.imshow("Gaussian", blurred_image)
cv2.imshow("Laplacian", laplacian_image)

cv2.waitKey(0)
