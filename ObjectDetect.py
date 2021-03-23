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


def detect_features(crop, mask=None):
    orb = cv2.ORB_create(nfeatures=700, scoreType=cv2.ORB_HARRIS_SCORE)
    img_copy = crop.copy()
    roi = img_copy

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    kp1, des1 = orb.detectAndCompute(img_copy, mask)
    return [kp1, des1, roi_hist, img_copy]


cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_select)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
FlannMatcher = cv2.FlannBasedMatcher(index_params, search_params)
draw_params = dict(matchColor = (0, 255, 0), singlePointColor=None, flags=2)



bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    _, img = cap.read()
    if select == True and cropped == False and redraw == False:
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("image", img)
    elif select == False and cropped == True and redraw == False:
        kp1, des1, roi_hist, img_copy = detect_features(img[y_start:y_end, x_start:x_end])
        print(img[y_start:y_end, x_start:x_end])
        redraw = True
        cropped = False

    if redraw:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv_img], [0, 1], roi_hist, [0, 180, 0, 256], 1)
       #_, mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
        kp2, des2, _, _ = detect_features(img, mask)

        if len(kp2) > 1:
            des1 = np.float32(des1)
            des2 = np.float32(des2)
            matches = FlannMatcher.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.65* n.distance:
                    good.append([m])
            matching_result = cv2.drawMatchesKnn(img_copy, kp1, img, kp2, good, None,**draw_params)
            cv2.imshow("result", matching_result)

        else:
            redraw = False
            cv2.destroyWindow("result")
            cv2.imshow("image", img)
    else:
        cv2.imshow("image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
