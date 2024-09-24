import cv2

#load the image
pict = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

#initialize the SIFT detector
sift = cv2.SIFT_create()

#detect keypoints and compute descriptors for the image
keypoints_pict, descriptors_pict = sift.detectAndCompute(pict, None)

#create a FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

#initialize cam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    #debugging
    print("Error: Could not open camera.")
    exit()

#loop to keep the webcam active
while True:
    #capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        #debugging
        print("Failed to capture image")
        break
    
    #convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect keypoints and compute descriptors for the frame
    keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)
    #(debugging)check if descriptors are available
    if descriptors_frame is None:
        continue
    
    #match descriptors between the  image and the frame
    matches = flann.knnMatch(descriptors_pict, descriptors_frame, k=2)
    
    #store all the good matches using lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    #draw matches
    result = cv2.drawMatches(pict, keypoints_pict, frame_gray, keypoints_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    #show the result
    cv2.imshow('Live SIFT Matches', result)

    #break the loop (end detection) if the window is closed
    if cv2.waitKey(1) & (int)(cv2.getWindowProperty('Live SIFT Matches', cv2.WND_PROP_VISIBLE)) < 1:
        break
    #also why does getWindowProperty function return as a float??

#release camera and clean up
cap.release()
cv2.destroyAllWindows()
