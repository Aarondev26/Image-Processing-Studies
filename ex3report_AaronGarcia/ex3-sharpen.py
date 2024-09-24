import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load input image
img_og = cv2.imread('coins.jpg')
img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)
width, height = img_og.shape
print(f'Image size: width={width}, height={height}')


img_og = cv2.normalize(img_og, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

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
mask = np.zeros((m,m),dtype=float)

# compute the center index
center = int(m / 2)

#for this example, m must be 3 (because although i attempted it, implementing the laplacian for general use was far too time consuming)
while m != 3:
    m = 3

# all comments are the corners of the s
# mask[0, 0] = 1
mask[1, 0] = 1
# mask[2, 0] = 1
mask[0, 1] = 1
mask[center, center] = -4 #-8
mask[1, 2] = 1
# mask[0, 2] = 1
mask[2, 1] = 1
# mask[2, 2] = 1

print(mask)

# create the laplacian version of the image
img_laplace = np.zeros((width, height))
# normalize it
img_laplace = cv2.normalize(img_laplace, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Main Algorithm:
for x in range(int(m/2), width-int(m/2)):
    for y in range(int(m/2), height-int(m/2)):
        value = 0
        # iterate through the mask array which is "overlayed" on the image
        for i in range(-int(m/2), int(m/2)+1):
            for j in range(-int(m/2), int(m/2)+1):
                # multiply the overlain mask values onto the pixel locations of the image
                value += (img_og[x+i, y+j] * mask[i+1, j+1])
        
        # place results at the pixel location in the new image.
        img_laplace[x, y] = (value)

# ERROR CORRECTION: normalize values in between 0-255 to fit in 8-bit int, otherwise the image output is not correct.

# create a placeholder valeiable to hole the scaled version of the laplacian image
img_scale = np.zeros((width, height), dtype=float)

img_min = np.zeros((width, height), dtype=float)

# enact the equation presented on the slides
# find the min of all scale values in the laplacian image
min = np.min(img_laplace)
for x in range(int(m/2), width-int(m/2)):
    for y in range(int(m/2), height-int(m/2)):
        # for all pixels, subtract the gray value by the minimum
        img_min[x,y] = img_laplace[x,y] - min

# scale the result
img_scale = 255*(img_min/np.max(img_min))

# create the final image placeholder
img_final = np.zeros((width, height), dtype=float)

# subtract the original by the laplacian to create the final sharpened output 
img_final = img_og + (-img_scale)

# normalize to display
img_final = cv2.normalize(img_final, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# normalize to display
img_og = cv2.normalize(img_og, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# show before and after.
cv2.imshow("Original", img_og)
cv2.imshow("Filtered", img_final)

cv2.waitKey(0)