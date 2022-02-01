'''
Project's Idea: Simple image classifier using TFLite with a Raspberry Pi Car model with PiCam.
Note: This example with OpenCV and PIL libraries.
'''
'''
Download command for TensorFlow Lite on Raspberry Pi OS [Linux]:

python3 -m pip install tflite-runtime
'''

import time
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter
import YB_Pcb_Car  # Import Yahboom car library
import cv2
car = YB_Pcb_Car.YB_Pcb_Car()
cap = cv2.VideoCapture(0)


def look_to_object():
    print('Start Look to object')
    car.Ctrl_Servo(2, 160)
    car.Ctrl_Servo(1, 90)
    time.sleep(1)


def say_cup():
    print('Say Cup')
    car.Ctrl_Servo(1, 160)
    time.sleep(0.5)
    car.Ctrl_Servo(1, 20)
    time.sleep(0.5)
    car.Car_Stop()


def say_card():
    print('Say Card')
    car.Ctrl_Servo(2, 0)
    time.sleep(0.5)
    car.Ctrl_Servo(2, 130)
    time.sleep(0.5)
    car.Ctrl_Servo(2, 30)
    time.sleep(0.5)
    car.Car_Stop()


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def main():

    interpreter = Interpreter('model.tflite')
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    look_to_object()
    while True:
        _, frame = cap.read()
        cv2.imshow('PiCam', frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        k = cv2.waitKey(1)
        if k == ord('a'):
            cv2.imwrite("frame.jpg", image)
            img_path = "frame.jpg"
            image = Image.open(img_path).resize(
                (width, height), Image.ANTIALIAS)
            results = classify_image(interpreter, image)
            label_id, prob = results[0]
            if label_id == 0:
                say_cup()
            elif label_id == 1:
                say_card()
            look_to_object()
            time.sleep(2)
        if k == ord('q'):
            break


if __name__ == '__main__':
    main()
    cap.release()
