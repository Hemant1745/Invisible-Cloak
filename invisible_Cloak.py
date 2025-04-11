import numpy as np
import cv2
import time

# To use webcam, enter 0. For video file, replace with path in double quotes
cap = cv2.VideoCapture(0)

time.sleep(3)  # Camera needs time to adjust according to the environment

background = 0

# Capturing the background for 60 frames
for i in range(60):
    ret, background = cap.read()

# Flip the background for mirror effect
background = np.flip(background, axis=1)

while cap.isOpened():  # This will run only when the webcam is open
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)
    
    # Convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Setting the values for black cloak
    lower_black = np.array([0, 0, 0])  # Low saturation and low value (brightness)
    upper_black = np.array([180, 255, 50])  # Max hue, max saturation, and low value

    # Create a mask for black areas
    mask1 = cv2.inRange(hsv, lower_black, upper_black)

    # Clean the mask using morphological operations
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

    # Invert the mask to get non-black parts
    mask2 = cv2.bitwise_not(mask1)

    # Segment out the black cloak part and the rest of the image
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)

    # Merge both results to get the final output
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Show the final result
    cv2.imshow('Invisible Cloak', final_output)

    # Exit if ESC key is pressed
    k = cv2.waitKey(10)
    if k == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
