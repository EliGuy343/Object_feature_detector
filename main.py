import cv2
import numpy as np

select = False
cropped = False
redraw = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cap = cv2.VideoCapture(0)


def mouse_select(event, x, y, flags, params):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, select, cropped

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        select = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if select == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        select = False  # cropping is finished
        if (x_end - x_start > 2) and (y_end - y_start > 2):
            cropped = True


def detect_features(crop):
    orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_HARRIS_SCORE)
    img_copy = crop.copy()
    roi = img_copy
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    kp1, des1 = orb.detectAndCompute(img_copy, None)
    return [kp1, des1, img_copy]


cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_select)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    _, img = cap.read()
    if select == True and cropped == False and redraw == False:
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("image", img)
    elif select == False and cropped == True and redraw == False:
        kp1, des1, img_copy = detect_features(img[y_start:y_end, x_start:x_end])
        redraw = True
        Cropped = False

    if redraw:
        kp2, des2, _ = detect_features(img)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matching_result = cv2.drawMatches(img_copy, kp1, img, kp2, matches, None)
        cv2.imshow("result", matching_result)
    else:
        cv2.imshow("image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
