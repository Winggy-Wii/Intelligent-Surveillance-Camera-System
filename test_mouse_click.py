import cv2
import numpy as np
img = np.zeros((512, 512, 3), np.uint8)


def get_mouse_coor(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y


cv2.namedWindow('image')
cv2.setMouseCallback('image', get_mouse_coor)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX, mouseY)
