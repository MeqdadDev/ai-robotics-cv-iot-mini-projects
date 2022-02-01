'''
Project's Idea: Simple image classifier using TFLite with a Raspberry Pi Car model with PiCam.
Note: This example with picamera library, without using OpenCV.
'''

'''
Download command for TensorFlow Lite on Raspberry Pi OS [Linux]:

python3 -m pip install tflite-runtime
'''


import io
import time
import numpy as np
import picamera
from PIL import Image
from tflite_runtime.interpreter import Interpreter
import YB_Pcb_Car  # Import Raspberry Pi car library
car = YB_Pcb_Car.YB_Pcb_Car()


def look_to_object():
    print('Start Look to object')
    # Vertically
    car.Ctrl_Servo(2, 160)
    car.Ctrl_Servo(1, 90)
    time.sleep(1)
    print('End of Look to object')


def say_no():
    car.Ctrl_Servo(1, 160)
    time.sleep(0.5)
    car.Ctrl_Servo(1, 44)
    time.sleep(0.5)
    car.Ctrl_Servo(1, 140)
    time.sleep(0.5)
    car.Ctrl_Servo(1, 30)
    car.Car_Spin_Left(70, 70)
    time.sleep(1)
    car.Car_Spin_Right(70, 70)
    time.sleep(1)

    car.Car_Stop()


def say_yes():
    print('Say Yes')
    car.Ctrl_Servo(2, 0)
    time.sleep(0.5)
    car.Ctrl_Servo(2, 130)
    time.sleep(0.5)
    car.Ctrl_Servo(2, 30)
    time.sleep(0.5)
    car.Ctrl_Servo(2, 130)
    time.sleep(0.5)
    car.Car_Back(60, 60)
    time.sleep(1)
    car.Car_Run(60, 60)
    time.sleep(1)
    car.Car_Stop()


def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


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

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def main():

    labels = load_labels('labels.txt')

    interpreter = Interpreter('model.tflite')
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
        camera.start_preview()
        look_to_object()
        try:
            stream = io.BytesIO()
            for _ in camera.capture_continuous(
                    stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                image = Image.open(stream).convert('RGB').resize((width, height),
                                                                 Image.ANTIALIAS)
                start_time = time.time()
                results = classify_image(interpreter, image)
                elapsed_ms = (time.time() - start_time) * 1000
                label_id, prob = results[0]
                print(label_id)
                if label_id == 0:
                    say_no()
                elif label_id == 1:
                    say_yes()
                camera.annotate_text = '%s %.2f\n%.1fms' % (
                    labels[label_id], prob, elapsed_ms)
                look_to_object()
                time.sleep(2)
                stream.seek(0)
                stream.truncate()
        finally:
            camera.stop_preview()


if __name__ == '__main__':
    main()
