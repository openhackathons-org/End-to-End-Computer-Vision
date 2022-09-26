import cv2
import imutils
import numpy as np


def calc_object_size(img_path, output_path="output", pixels_per_metric=38):
    
    # load image with OpenCV and blur it slightly
    image = cv2.imread(img_path)  
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # convert to hsv color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # color thresholding
    # orange color range values: Hue (5 - 25)
    lower = np.array([5, 130, 155])
    upper = np.array([25, 255, 255])
    # using inRange function to get only orange colors
    mask = cv2.inRange(hsv, lower, upper)
    # remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # find contours in the edge map
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours and keep the largest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    c = cnts[0]

    # compute the minimum enclosing circle of the contour
    orig = image.copy()
    (x, y), radius = cv2.minEnclosingCircle(c)

    # draw the circle
    cv2.circle(orig, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    # draw a diameter and end points
    cv2.line(orig, (int(x - radius), int(y)), (int(x + radius), int(y)),
        (255, 0, 255), 2)
    cv2.circle(orig, (int(x - radius), int(y)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(x + radius), int(y)), 5, (255, 0, 0), -1)

    # draw the center
    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # compute the size of the object
    dimR = radius / pixels_per_metric
    print(f"Diameter of the object: {2 * dimR:.1f}cm")

    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}cm".format(2 * dimR),
        (int(x - 15), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 0), 2)
    
    # save the output image
    cv2.imwrite(output_path, orig)

    return 2 * dimR
