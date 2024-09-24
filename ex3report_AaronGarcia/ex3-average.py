import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load input image
img_og = cv2.imread('test03.jpg')
img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)
width, height = img_og.shape
print(f'Image size: width={width}, height={height}')

# "m" represents the value of the size of the mask, and directly effects the blurring effect of the filter.
# as m increases, images gets more blurred
m = 3

# ERROR PREVENTION: if m is even, it will be changed to an odd number to ensure the following works properly. 
if m % 2 == 0:
    print("Inputted \"m\" = " + str(m) + " is not valid, reverting, using \"m\" = " + str(m-1) + "instead. ")
    m = m-1

# get square size
m_sqrd = m * m

# create 2D mask array 
mask = np.ones((m,m),dtype=int)
mask = mask / m_sqrd

# create the canvas for the filtered image
img_filtered = np.zeros([width, height])

# Main Algorithm:
for x in range(int(m/2), width-int(m/2)):
    for y in range(int(m/2), height-int(m/2)):
        avg = 0

        # iterate through the mask array
        for i in range(-int(m/2), int(m/2)+1):
            for j in range(-int(m/2), int(m/2)+1):
                
                # generate the average by iterating through the mask array and the img simulataneously,
                # multiplying and adding each result to a a total pot. (this will require us to normalize later)
                avg = avg + img_og[x+i, y+j]*mask[i+1, j+1]
        
        # place result at the pixel location in the new image.
        img_filtered[x, y] = (avg)

# ERROR CORRECTION: normalize values in between 0-255 to fit in 8-bit int, otherwise the image output is not correct.
img_filtered = cv2.normalize(img_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# show before and after.
cv2.imshow("Original", img_og)
cv2.imshow("Filtered", img_filtered)

cv2.waitKey(0)