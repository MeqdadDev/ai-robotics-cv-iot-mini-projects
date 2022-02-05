'''
IoT Mini Project
Weather Station using DHT Sensor and Raspberry Pi with ThingSpeak Platform
Code Sample: Interfacing DHT22 with Raspberry Pi and sending the data to an IoT Platform (ThingSpeak Platform)
'''

from time import sleep
# import Adafruit_DHT # Not supported library
import adafruit_dht
from board import *
import requests


# After creating your account on ThingSpeak platform, put your channel id below
channel_id = 12345
write_key = 'WriteYourKeyAsString.......'  # Put your write key here

# D4 = GPIO4 / D17 = GPIO17 ...etc.
SENSOR_PIN = D17


def get_measurements():
    dht22 = adafruit_dht.DHT22(SENSOR_PIN, use_pulseio=False)
    temperature = dht22.temperature
    humidity = dht22.humidity

    print(f"Humidity= {humidity:.2f}")
    print(f"Temperature= {temperature:.2f}°C")

    return temperature, humidity


def sendData(temp, humidity):
	url = 'https://api.thingspeak.com/update'
    params = {'key': write_key, 'field1': temp, 'field2': humidity}
	res = requests.get(url, params=params)

if __name__ == "__main__":
    while True:
        # 15 seconds is the minimum time for the free account on ThingSpeak
        sleep(15)
        try:        
            temperature, humidity = get_measurements()
        except:
            print("Error: Can't get the sensor values, check out your wiring connection.")
        try:
            sendData(temperature, humidity)
        except:
            print("Error: Can't push the sensor values to ThingSpeak server.")
