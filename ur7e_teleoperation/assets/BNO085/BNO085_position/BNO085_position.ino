#include <micro_ros_arduino.h>
#include <WiFi.h>

#include <stdio.h>
#include <math.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rmw_microros/rmw_microros.h>

#include <sensor_msgs/msg/imu.h>
#include <geometry_msgs/msg/point.h>

#include <Adafruit_BNO08x.h>

// ── BNO08x setup ──────────────────────────────────────────────────────
#define BNO08X_RESET -1
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// ── micro-ROS objects ─────────────────────────────────────────────────
rcl_publisher_t           publisher;      // imu/data -- orientation + raw accel, unchanged
rcl_publisher_t           pos_publisher;  // probe/position -- Kalman+ZUPT filtered position
sensor_msgs__msg__Imu     imu_msg;
geometry_msgs__msg__Point pos_msg;
rclc_support_t            support;
rcl_allocator_t           allocator;
rcl_node_t                node;

#define LED_PIN 2

#define RCCHECK(fn) { \
  rcl_ret_t temp_rc = fn; \
  if ((temp_rc != RCL_RET_OK)) { \
    Serial.print("[RCCHECK FAIL] line "); Serial.print(__LINE__); \
    Serial.print(" rc="); Serial.println((int)temp_rc); \
    error_loop(); \
  } \
}
#define RCSOFTCHECK(fn) { \
  rcl_ret_t temp_rc = fn; \
  if ((temp_rc != RCL_RET_OK)) { \
    Serial.print("[RCSOFTCHECK FAIL] line "); Serial.print(__LINE__); \
    Serial.print(" rc="); Serial.println((int)temp_rc); \
  } \
}

enum AgentState {
  WAITING_AGENT,
  AGENT_AVAILABLE,
  AGENT_CONNECTED,
  AGENT_DISCONNECTED
};
AgentState agent_state = WAITING_AGENT;

unsigned long last_ping_time = 0;
#define PING_TIMEOUT_MS 200

// ══════════════════════════════════════════════════════════════════════
// Kalman filter (per-axis, state = [position, velocity]) + ZUPT
// ══════════════════════════════════════════════════════════════════════
struct KalmanAxis1D {
  float pos;
  float vel;
  float P[2][2];
};

// Tune these against YOUR sensor's real noise floor. Log raw accel at
// rest first (watch the imu/data linear_acceleration field) before
// trusting these defaults.
const float KALMAN_ACCEL_NOISE_STD    = 0.03f;   // m/s^2 -- trust in accel signal
const float KALMAN_ZUPT_VEL_NOISE_STD = 0.01f;  // m/s   -- how firmly ZUPT snaps vel to 0

void kalman_init(KalmanAxis1D &kf) {
  kf.pos = 0.0f;
  kf.vel = 0.0f;
  kf.P[0][0] = 0.01f; kf.P[0][1] = 0.0f;
  kf.P[1][0] = 0.0f;  kf.P[1][1] = 0.01f;
}

void kalman_predict(KalmanAxis1D &kf, float accel, float dt) {
  if (dt <= 0.0f) return;

  // State transition: pos' = pos + vel*dt + 0.5*a*dt^2, vel' = vel + a*dt
  float pos_new = kf.pos + kf.vel * dt + 0.5f * accel * dt * dt;
  float vel_new = kf.vel + accel * dt;

  // F = [[1, dt], [0, 1]] -- propagate covariance P_pred = F P F^T + Q
  float P00 = kf.P[0][0], P01 = kf.P[0][1];
  float P10 = kf.P[1][0], P11 = kf.P[1][1];

  float FP00 = P00 + dt * P10;
  float FP01 = P01 + dt * P11;
  float FP10 = P10;
  float FP11 = P11;

  float newP00 = FP00 + dt * FP01;
  float newP01 = FP01;
  float newP10 = FP10 + dt * FP11;
  float newP11 = FP11;

  float q   = KALMAN_ACCEL_NOISE_STD * KALMAN_ACCEL_NOISE_STD;
  float dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
  float Q00 = q * 0.25f * dt4;
  float Q01 = q * 0.5f  * dt3;
  float Q11 = q * dt2;

  kf.pos = pos_new;
  kf.vel = vel_new;
  kf.P[0][0] = newP00 + Q00;
  kf.P[0][1] = newP01 + Q01;
  kf.P[1][0] = newP10 + Q01;   // symmetric
  kf.P[1][1] = newP11 + Q11;
}

void kalman_correct_zupt(KalmanAxis1D &kf) {
  // Measurement: "velocity is 0 right now", trusted strongly (small R).
  float R = KALMAN_ZUPT_VEL_NOISE_STD * KALMAN_ZUPT_VEL_NOISE_STD;

  float y = 0.0f - kf.vel;         // innovation
  float S = kf.P[1][1] + R;        // innovation covariance (H picks out P11)
  float K0 = kf.P[0][1] / S;       // gain applied to position
  float K1 = kf.P[1][1] / S;       // gain applied to velocity

  kf.pos = kf.pos + K0 * y;
  kf.vel = kf.vel + K1 * y;

  float P00 = kf.P[0][0], P01 = kf.P[0][1];
  float P10 = kf.P[1][0], P11 = kf.P[1][1];

  kf.P[0][0] = P00 - K0 * P10;
  kf.P[0][1] = P01 - K0 * P11;
  kf.P[1][0] = P10 - K1 * P10;
  kf.P[1][1] = P11 - K1 * P11;
}

KalmanAxis1D kf_x, kf_y, kf_z;

// ── Stationary detector: sliding window of accel magnitude ───────────────
#define STATIONARY_WINDOW_SIZE 25
float accel_mag_window[STATIONARY_WINDOW_SIZE];
int   accel_mag_window_idx   = 0;
int   accel_mag_window_count = 0;

// Tune against your real noise floor (log accel at rest first).
const float STATIONARY_MAG_THRESHOLD = 0.1f;  // m/s^2
const float STATIONARY_VAR_THRESHOLD = 0.03f;

bool stationary_update(float mag) {
  accel_mag_window[accel_mag_window_idx] = mag;
  accel_mag_window_idx = (accel_mag_window_idx + 1) % STATIONARY_WINDOW_SIZE;
  if (accel_mag_window_count < STATIONARY_WINDOW_SIZE) accel_mag_window_count++;

  if (accel_mag_window_count < STATIONARY_WINDOW_SIZE) return false;

  float mean = 0.0f;
  for (int i = 0; i < STATIONARY_WINDOW_SIZE; i++) mean += accel_mag_window[i];
  mean /= STATIONARY_WINDOW_SIZE;

  float var = 0.0f;
  for (int i = 0; i < STATIONARY_WINDOW_SIZE; i++) {
    float d = accel_mag_window[i] - mean;
    var += d * d;
  }
  var /= STATIONARY_WINDOW_SIZE;

  return (mean < STATIONARY_MAG_THRESHOLD) && (var < STATIONARY_VAR_THRESHOLD);
}

unsigned long last_accel_micros = 0;

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

bool create_entities() {
  allocator = rcl_get_default_allocator();

  if (rclc_support_init(&support, 0, NULL, &allocator) != RCL_RET_OK) return false;
  if (rclc_node_init_default(&node, "bno08x_imu_node", "", &support) != RCL_RET_OK) return false;

  if (rclc_publisher_init(
        &publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, Imu),
        "imu/data",
        &rmw_qos_profile_sensor_data) != RCL_RET_OK) return false;

  if (rclc_publisher_init(
        &pos_publisher,
        &node,
        ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point),
        "probe/position",
        &rmw_qos_profile_sensor_data) != RCL_RET_OK) return false;

  imu_msg.header.frame_id.data     = (char *) "imu_link";
  imu_msg.header.frame_id.size     = strlen("imu_link");
  imu_msg.header.frame_id.capacity = strlen("imu_link") + 1;

  memset(imu_msg.orientation_covariance, 0, sizeof(imu_msg.orientation_covariance));
  memset(imu_msg.linear_acceleration_covariance, 0, sizeof(imu_msg.linear_acceleration_covariance));
  imu_msg.angular_velocity_covariance[0] = -1;

  // NOTE: Kalman filter state is intentionally NOT reset here. This
  // function re-runs on every agent reconnect -- resetting kf_x/y/z here
  // would silently snap tracked position back to zero on every WiFi
  // hiccup. It's initialized once in setup() instead.

  return true;
}

void destroy_entities() {
  rcl_publisher_fini(&publisher, &node);
  rcl_publisher_fini(&pos_publisher, &node);
  rcl_node_fini(&node);
  rclc_support_fini(&support);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  Wire.begin();
  if (!bno08x.begin_I2C(0x4B)) {
    Serial.println("Failed to find BNO08x chip");
    error_loop();
  }
  Serial.println("BNO08x Found!");
  setReports();

  // Kalman filter + stationary detector init -- once, at boot, not tied
  // to the micro-ROS entity lifecycle (see note in create_entities()).
  kalman_init(kf_x);
  kalman_init(kf_y);
  kalman_init(kf_z);
  accel_mag_window_idx   = 0;
  accel_mag_window_count = 0;
  last_accel_micros      = 0;

  Serial.println("connecting wifi transport...");
  // set_microros_wifi_transports("dev", "123456789", "10.42.0.1", 8888);
  // set_microros_wifi_transports("OPPO", "123456789", "10.42.0.1", 8888);
  set_microros_transports();

  // Disable WiFi power-save (modem sleep) -- avoids recurring 100-300ms+
  // stalls on the published topics.
  WiFi.setSleep(false);

  Serial.println("Setup complete, entering loop()");
}

void loop() {
  switch (agent_state) {
    case WAITING_AGENT:
      if (millis() - last_ping_time > PING_TIMEOUT_MS) {
        last_ping_time = millis();
        agent_state = (RMW_RET_OK == rmw_uros_ping_agent(100, 1)) ? AGENT_AVAILABLE : WAITING_AGENT;
      }
      break;

    case AGENT_AVAILABLE:
      if (create_entities()) {
        Serial.println("[IMU] Agent connected, entities created");
        agent_state = AGENT_CONNECTED;
      } else {
        destroy_entities();
        agent_state = WAITING_AGENT;
      }
      break;

    case AGENT_CONNECTED:
      if (millis() - last_ping_time > PING_TIMEOUT_MS) {
        last_ping_time = millis();
        agent_state = (RMW_RET_OK == rmw_uros_ping_agent(100, 1)) ? AGENT_CONNECTED : AGENT_DISCONNECTED;
      }

      if (agent_state == AGENT_CONNECTED) {
        if (bno08x.wasReset()) {
          Serial.println("Sensor was reset, re-enabling reports");
          setReports();
        }

        if (bno08x.getSensorEvent(&sensorValue)) {
          bool have_new_data = false;

          switch (sensorValue.sensorId) {
            case SH2_ROTATION_VECTOR:
              imu_msg.orientation.x = sensorValue.un.rotationVector.i;
              imu_msg.orientation.y = sensorValue.un.rotationVector.j;
              imu_msg.orientation.z = sensorValue.un.rotationVector.k;
              imu_msg.orientation.w = sensorValue.un.rotationVector.real;
              have_new_data = true;
              break;

            case SH2_LINEAR_ACCELERATION: {
              float ax = sensorValue.un.linearAcceleration.x;
              float ay = sensorValue.un.linearAcceleration.y;
              float az = sensorValue.un.linearAcceleration.z;

              imu_msg.linear_acceleration.x = ax;
              imu_msg.linear_acceleration.y = ay;
              imu_msg.linear_acceleration.z = az;
              have_new_data = true;

              // Rotate raw accel into the EE-aligned frame -- fixed 90 deg
              // rotation about X, matching the R_CALIB previously applied
              // on the Python side. Done here now so Python no longer
              // needs to do any frame math on translation.
              float ax_ee =  ax;
              float ay_ee =  ay;
              float az_ee =  az;

              unsigned long now_us = micros();
              float dt = (last_accel_micros == 0) ? 0.0f
                         : (now_us - last_accel_micros) / 1000000.0f;
              last_accel_micros = now_us;

              float mag = sqrtf(ax_ee*ax_ee + ay_ee*ay_ee + az_ee*az_ee);
              bool is_stationary = stationary_update(mag);

              if (dt > 0.0f) {
                kalman_predict(kf_x, ax_ee, dt);
                kalman_predict(kf_y, ay_ee, dt);
                kalman_predict(kf_z, az_ee, dt);
              }

              if (is_stationary) {
                kalman_correct_zupt(kf_x);
                kalman_correct_zupt(kf_y);
                kalman_correct_zupt(kf_z);
              }

              pos_msg.x = (double)kf_x.pos;
              pos_msg.y = (double)kf_y.pos;
              pos_msg.z = (double)kf_z.pos;

              rcl_ret_t pos_pub_rc = rcl_publish(&pos_publisher, &pos_msg, NULL);
              if (pos_pub_rc != RCL_RET_OK) {
                Serial.print("position publish failed, rc=");
                Serial.println(pos_pub_rc);
              }
              break;
            }

            default:
              break;
          }

          if (have_new_data) {
            int64_t now_ms = rmw_uros_epoch_millis();
            imu_msg.header.stamp.sec     = (int32_t)(now_ms / 1000);
            imu_msg.header.stamp.nanosec = (uint32_t)((now_ms % 1000) * 1000000);

            rcl_ret_t pub_rc = rcl_publish(&publisher, &imu_msg, NULL);
            if (pub_rc != RCL_RET_OK) {
              Serial.print("publish failed, rc=");
              Serial.println(pub_rc);
            }
          }
        }
      }
      break;

    case AGENT_DISCONNECTED:
      Serial.println("[IMU] Agent disconnected, tearing down and retrying");
      destroy_entities();
      agent_state = WAITING_AGENT;
      break;

    default:
      break;
  }

  digitalWrite(LED_PIN, agent_state == AGENT_CONNECTED ? LOW : HIGH);
}