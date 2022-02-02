'''
Notes before you start:
1- Create a folder called 'CollectedData'.
2- Write the person's name in the name variable in the code below.
3- Inside 'CollectedData' folder, create another folder with the person's name.
4- Repeat these steps for each person to be recognised in the system.
'''
import cv2 as cv
import os

# To get the file's path
path = os.getcwd()

# Your name here....
name = 'Meqdad'

cam = cv.VideoCapture(0)
img_counter = 0

while True:
    is_success, frame = cam.read()
    if not is_success:
        print("failed to capture image.")
        break
    cv.imshow("Collecting Data Phase | Press space to capture", frame)

    # Stores the pressed key by the user.
    k = cv.waitKey(1)

    # Check if ESC is pressed
    if k % 256 == 27:
        print("ESC is pressed... Cancel")
        break

    # Check if space is pressed
    elif k % 256 == 32:
        img_name = f"{path}/CollectedData/{name}/{name}_img_{img_counter}.jpg"
        cv.imwrite(img_name, frame)
        print(f"Image #{img_counter} saved!")
        img_counter += 1

cam.release()
cv.destroyAllWindows()
