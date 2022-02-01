###****************** Example 2 ******************###
# Extracting characters from webcam/PiCam.
###***********************************************###

import cv2 as cv
import pytesseract

cam = cv.VideoCapture(0)


def get_ocr(img):
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return pytesseract.image_to_string(rgb)


def screenshot(img_frame):
    return cv.imwrite('Screenshot.jpg', img_frame)


while True:
    is_success, frame = cam.read()
    if not is_success:
        print("failed to capture image.")
        break
    cv.imshow("OCR Stream", frame)

    k = cv.waitKey(1)

    # Check if space is pressed
    if k % 256 == 32:
        screenshot(frame)
        print("Image captured!")
        image = cv.imread('Screenshot.jpg')
        print(get_ocr(image))

    # Check if ESC is pressed
    if k % 256 == 27:
        print("ESC is pressed...")
        break

cv.destroyAllWindows()
