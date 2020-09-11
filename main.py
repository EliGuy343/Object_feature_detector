import cv2
import numpy as np

select = False
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

        refPoint = [(x_start, y_start), (x_end, y_end)]




cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_select)

while True:
    _, img = cap.read()
    if select:
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("image", img)

    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
