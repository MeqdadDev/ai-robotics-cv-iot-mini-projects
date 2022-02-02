import cv2 as cv

cam = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_face(img):
    face_img = img.copy()

    # Returns the rectangle borders
    face_rects = face_cascade.detectMultiScale(face_img)

    for (x, y, w, h) in face_rects:
        cv.rectangle(face_img, (x, y), (x+w, y+h), (255, 255, 255), 10)
    return face_img


while True:
    _, frame = cam.read()
    cv.imshow("Show", frame)

    detected_face = detect_face(frame)
    cv.imshow('Final', detected_face)
    k = cv.waitKey(1)
    if k % 256 == 27:
        print("ESC...")
        break


cam.release()
cv.destroyAllWindows()
