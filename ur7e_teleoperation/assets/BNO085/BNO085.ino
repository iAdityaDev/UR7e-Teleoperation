#include <micro_ros_arduino.h>
#include <WiFi.h>

#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>

#include <sensor_msgs/msg/imu.h>

#include <Adafruit_BNO08x.h>

// ── BNO08x setup ──────────────────────────────────────────────────────
#define BNO08X_RESET -1
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// ── micro-ROS objects ─────────────────────────────────────────────────
// NOTE: no executor. We only publish (no subscriptions/timers/services),
// so rclc_executor was doing nothing for us except crashing setup --
// rclc_executor_init() requires at least 1 handle, and we were passing 0.
rcl_publisher_t       publisher;
sensor_msgs__msg__Imu imu_msg;
rclc_support_t        support;
rcl_allocator_t       allocator;
rcl_node_t            node;

#define LED_PIN 2

#define RCCHECK(fn) { \
  rcl_ret_t temp_rc = fn; \
  if ((temp_rc != RCL_RET_OK)) { \
    Serial.print("[RCCHECK FAIL] line "); Serial.print(__LINE__); \
    Serial.print(" rc="); Serial.println((int)temp_rc); \
    error_loop(); \
  } \
}

void error_loop() {
  while (1) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(100);
  }
}

void setReports() {
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 20000)) {
    Serial.println("Could not enable rotation vector");
  } else {
    Serial.println("Rotation vector report enabled");
  }

  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, 20000)) {
    Serial.println("Could not enable linear acceleration");
  } else {
    Serial.println("Linear acceleration report enabled");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_PIN, OUTPUT);

  Wire.begin();
  if (!bno08x.begin_I2C(0x4B)) {
    Serial.println("Failed to find BNO08x chip");
    error_loop();
  }
  Serial.println("BNO08x Found!");
  setReports();

  Serial.println("connecting wifi transport...");
  set_microros_wifi_transports("OPPO", "123456789", "10.131.2.181", 8888);
  Serial.println("wifi transport up");
  digitalWrite(LED_PIN, HIGH);   // ON while micro-ROS is initializing

  allocator = rcl_get_default_allocator();

  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "bno08x_imu_node", "", &support));

  RCCHECK(rclc_publisher_init(
    &publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, Imu),
    "imu/data",
    &rmw_qos_profile_sensor_data));

  imu_msg.header.frame_id.data     = (char *) "imu_link";
  imu_msg.header.frame_id.size     = strlen("imu_link");
  imu_msg.header.frame_id.capacity = strlen("imu_link") + 1;

  memset(imu_msg.orientation_covariance, 0, sizeof(imu_msg.orientation_covariance));
  memset(imu_msg.linear_acceleration_covariance, 0, sizeof(imu_msg.linear_acceleration_covariance));
  imu_msg.angular_velocity_covariance[0] = -1;

  Serial.println("Setup complete, entering loop()");
  digitalWrite(LED_PIN, LOW);   // OFF once ready
}

void loop() {
  if (bno08x.wasReset()) {
    Serial.println("Sensor was reset, re-enabling reports");
    setReports();
  }

  if (!bno08x.getSensorEvent(&sensorValue)) {
    return;
  }

  bool have_new_data = false;

  switch (sensorValue.sensorId) {
    case SH2_ROTATION_VECTOR:
      imu_msg.orientation.x = sensorValue.un.rotationVector.i;
      imu_msg.orientation.y = sensorValue.un.rotationVector.j;
      imu_msg.orientation.z = sensorValue.un.rotationVector.k;
      imu_msg.orientation.w = sensorValue.un.rotationVector.real;
      have_new_data = true;
      break;

    case SH2_LINEAR_ACCELERATION:
      imu_msg.linear_acceleration.x = sensorValue.un.linearAcceleration.x;
      imu_msg.linear_acceleration.y = sensorValue.un.linearAcceleration.y;
      imu_msg.linear_acceleration.z = sensorValue.un.linearAcceleration.z;
      have_new_data = true;
      break;

    default:
      break;
  }

  if (!have_new_data) return;

  int64_t now_ms = rmw_uros_epoch_millis();
  imu_msg.header.stamp.sec     = (int32_t)(now_ms / 1000);
  imu_msg.header.stamp.nanosec = (uint32_t)((now_ms % 1000) * 1000000);

  rcl_ret_t pub_rc = rcl_publish(&publisher, &imu_msg, NULL);
  if (pub_rc != RCL_RET_OK) {
    Serial.print("publish failed, rc=");
    Serial.println(pub_rc);
  }
}