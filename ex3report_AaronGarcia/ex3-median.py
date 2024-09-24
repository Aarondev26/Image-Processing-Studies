import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load input image
img_og = cv2.imread('lena-noise.jpg')
img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)
width, height = img_og.shape
print(f'Image size: width={width}, height={height}')

# "m" represents the value of the size of the mask, and directly effects the strength of the filter.
# as m increases, the image has less salt and pepper effect (until there is none), but has increased blobbing.
m = 3

# get square size
m_sqrd = m * m

# ERROR PREVENTION: if m is even, it will be changed to an odd number to ensure the following works properly. 
if m % 2 == 0:
    print("Inputted \"m\" = " + str(m) + " is not valid, reverting, using \"m\" = " + str(m-1) + "instead. ")
    m = m-1

# create the canvas for the filtered image
img_filtered = np.zeros([width, height])

# Main Algorithm:
for x in range(int(m/2), width-int(m/2)):
    for y in range(int(m/2), height-int(m/2)):
        values = []

        # iterate through the mask array which is "overlayed" on the image
        for i in range(-int(m/2), int(m/2)+1):
            for j in range(-int(m/2), int(m/2)+1):
                
                # gather all of the values of the pixels at the location of the mask.
                # (this implementation will require us to normalize later)
                values.append(img_og[x+i, y+j])
        
        # place result at the pixel location in the new image.
        values = sorted(values)
        # find the median, place it in the filtered image.
        img_filtered[x, y] = values[int(m_sqrd/2)]


# ERROR CORRECTION: normalize values in between 0-255 to fit in 8-bit int, otherwise the image output is not correct.
img_filtered = cv2.normalize(img_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# show before and after.
cv2.imshow("Original", img_og)
cv2.imshow("Filtered", img_filtered)

cv2.waitKey(0)