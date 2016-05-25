import serial
import time

ser = serial.Serial('/dev/cu.usbmodem1421')
filename = 'ref_' + str(time.time()) + '.txt'

while True:
  with open(filename, 'a') as f:
    f.write(ser.readline())