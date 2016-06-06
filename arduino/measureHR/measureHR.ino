
int sensorPin = A0;
int sensorValue = 0;
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
