import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load input image
img1 = cv2.imread('test02.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
height, width = img1.shape
print(f'Image size: width={width}, height={height}')

# Output image (blank)
img2 = np.zeros((height, width), np.uint8)

# Intensity Transformation
for y in range(height):
  for x in range(width):
    img2[y][x] = 255 - img1[y][x]

# Show input/output images
cv2.imshow('Input',img1)
cv2.imshow('Output',img2)
cv2.waitKey(0)
