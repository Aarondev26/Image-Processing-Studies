import cv2
import numpy as np
import matplotlib.pyplot as plt


def autocanny(image):
    #check if image is loaded successfully
    if image is None:
        print("Error: Could not open or find the image.")
        exit()

    #compute the median of the single-channel pixel intensities
    median_val = np.median(image)

    #apply automatic canny edge detection using the computed median
    lower_threshold = int(max(0, (1.0 - 0.33) * median_val))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_val))

    #apply canny edge detection
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

#read the image
image = cv2.imread('test51.jpg')

#convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#edge detection using canny, (taken from last assignment)
edges = autocanny(gray_image)

#hough line transform- harvest the generated line to be places on the final image
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100, None, 0, 0)

#draw the lines onto the original image
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

#display the results
#show before and after.
cv2.imshow("Original image", gray_image)
cv2.imshow("Edges image", edges)
cv2.imshow("Original+Lines image", image)

cv2.waitKey(0)
