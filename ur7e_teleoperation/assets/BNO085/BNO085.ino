#include <Wire.h>
#include <Adafruit_BNO08x.h>

#define BNO08X_RESET -1  // no hardware reset pin used

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

void setReports() {
  // Rotation vector gives you fused quaternion output
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR)) {
    Serial.println("Could not enable rotation vector");
  }
  // Also grab linear acceleration (gravity removed)
  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION)) {
    Serial.println("Could not enable linear acceleration");
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("Adafruit BNO08x test");

  if (!bno08x.begin_I2C()) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }
  Serial.println("BNO08x Found!");

  setReports();
  Serial.println("Reports enabled");
}

void loop() {
  if (bno08x.wasReset()) {
    Serial.println("Sensor was reset, re-enabling reports");
    setReports();
  }

  if (!bno08x.getSensorEvent(&sensorValue)) {
    return;
  }

  switch (sensorValue.sensorId) {
    case SH2_ROTATION_VECTOR:
      Serial.print("Quat: ");
      Serial.print("i="); Serial.print(sensorValue.un.rotationVector.i, 4);
      Serial.print(" j="); Serial.print(sensorValue.un.rotationVector.j, 4);
      Serial.print(" k="); Serial.print(sensorValue.un.rotationVector.k, 4);
      Serial.print(" real="); Serial.println(sensorValue.un.rotationVector.real, 4);
      break;

    case SH2_LINEAR_ACCELERATION:
      Serial.print("LinAccel: ");
      Serial.print("x="); Serial.print(sensorValue.un.linearAcceleration.x, 3);
      Serial.print(" y="); Serial.print(sensorValue.un.linearAcceleration.y, 3);
      Serial.print(" z="); Serial.println(sensorValue.un.linearAcceleration.z, 3);
      break;
  }
}