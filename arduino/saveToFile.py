import serial
ser = serial.Serial('/dev/cu.usbmodem1421')
while True:
  with open('true_HR.txt', 'a') as f:
    f.write(ser.readline())