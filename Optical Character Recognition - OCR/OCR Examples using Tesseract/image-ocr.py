
###****************** Example 1 ******************###
# Extracting characters from image.
###***********************************************###

import cv2 as cv
import pytesseract


def get_ocr(img):
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return pytesseract.image_to_string(rgb)


img = cv.imread('numbers.jpeg')

cv.imshow('Test OCR Image', img)
text_ocr = get_ocr(img)
print('Extracted Text using OCR: ', text_ocr)
cv.waitKey(0)
