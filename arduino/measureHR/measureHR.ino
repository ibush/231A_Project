
int sensorPin = A0;    // select the input pin for the potentiometer
int sensorValue = 0;  // variable to store the value coming from the sensor
uint32_t lastTimestamp = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
 for (;;) {
   const uint32_t now = micros();
   if (now-lastTimestamp >= 5000) { //200 sps
     lastTimestamp = now;
     break;
   }
  }
  sensorValue = analogRead(sensorPin);
  Serial.print(sensorValue);
  Serial.println();
}
