import genesis as gs
import numpy as np
import pyglet
from pyglet.window import key as pyglet_key
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque
import os
import glob
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu as ImuMsg
from geometry_msgs.msg import Point as PointMsg

gs.init(backend=gs.gpu)

# ── Live IMU input (BNO085 over micro-ROS) ───────────────────────────────
IMU_TOPIC = '/imu/data'          # orientation (+ raw accel, unused here now)
POS_TOPIC = '/probe/position'    # Kalman+ZUPT filtered position, computed on ESP32

class BNO085Subscriber(Node):
    """Subscribes to orientation AND the pre-filtered position topic.
    All Kalman/ZUPT math now happens on the ESP32 -- this class just
    exposes the latest values thread-safely, same pattern as before."""

    def __init__(self, imu_topic, pos_topic):
        super().__init__('genesis_teleop_imu_subscriber')
        self._lock = threading.Lock()

        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._has_quat = False
        self._last_quat_time = None

        self._pos = np.array([0.0, 0.0, 0.0])
        self._has_pos = False
        self._last_pos_time = None

        self.create_subscription(ImuMsg, imu_topic, self._imu_callback, qos_profile_sensor_data)
        self.create_subscription(PointMsg, pos_topic, self._pos_callback, qos_profile_sensor_data)
        self.get_logger().info(f'Subscribed to {imu_topic} and {pos_topic}, waiting for data...')

    def _imu_callback(self, msg):
        if len(msg.orientation_covariance) and msg.orientation_covariance[0] == -1.0:
            return  # driver is reporting "orientation not available"
        q = msg.orientation
        with self._lock:
            self._quat = np.array([q.w, q.x, q.y, q.z])
            self._has_quat = True
            self._last_quat_time = time.time()

    def _pos_callback(self, msg):
        with self._lock:
            self._pos = np.array([msg.x, msg.y, msg.z])
            self._has_pos = True
            self._last_pos_time = time.time()

    def get_quat(self):
        with self._lock:
            return self._quat.copy(), self._has_quat

    def get_pos(self):
        with self._lock:
            return self._pos.copy(), self._has_pos

    def data_age(self):
        """Seconds since the last orientation message arrived."""
        with self._lock:
            if self._last_quat_time is None:
                return float('inf')
            return time.time() - self._last_quat_time

    def pos_data_age(self):
        """Seconds since the last position message arrived."""
        with self._lock:
            if self._last_pos_time is None:
                return float('inf')
            return time.time() - self._last_pos_time


rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
imu_node = BNO085Subscriber(IMU_TOPIC, POS_TOPIC)
threading.Thread(target=rclpy.spin, args=(imu_node,), daemon=True).start()

scene = gs.Scene(
    profiling_options=gs.options.ProfilingOptions(
        show_FPS=False,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane())

table = scene.add_entity(
    gs.morphs.Box(size=(1.0, 1.0, 0.02), pos=(-0.3, 0.0, 0.5), fixed=True)
)

leg_height, leg_size = 0.5, 0.05
for pos in [
    (-0.3+0.45,  0.4, leg_height/2),
    (-0.3+0.45, -0.4, leg_height/2),
    (-0.3-0.45,  0.4, leg_height/2),
    (-0.3-0.45, -0.4, leg_height/2),
]:
    scene.add_entity(gs.morphs.Box(size=(leg_size, leg_size, leg_height), pos=pos, fixed=True))

ur5e = scene.add_entity(
    gs.morphs.URDF(
        file='/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/ur5e.urdf',
        fixed=True,
        pos=(0.0, 0.0, 0.5),
        links_to_keep=['probe_link'],
    )
)

imu_entity = scene.add_entity(
    gs.morphs.Box(size=(0.06, 0.04, 0.01), pos=(1.5, 0.0, 1.0), euler=(90, 180, 90), fixed=True)
)

human = scene.add_entity(
    gs.morphs.URDF(
        file='/home/deviant/human-model-generator/code/models/humanModels/kevin_ultrasound.urdf',
        pos=(0.6, 0.0, 0.55),
        euler=(0, 270, 90),
        fixed=True,
    )
)

table2 = scene.add_entity(
    gs.morphs.Box(size=(0.7, 1.5, 0.02), pos=(0.7, 0.0, 0.4), fixed=True)
)

leg_height2, leg_size2 = 0.4, 0.05
for pos in [
    (0.7+0.3,  0.7, leg_height2/2),
    (0.7+0.3, -0.7, leg_height2/2),
    (0.7-0.3,  0.7, leg_height2/2),
    (0.7-0.3, -0.7, leg_height2/2),
]:
    scene.add_entity(gs.morphs.Box(size=(leg_size2, leg_size2, leg_height2), pos=pos, fixed=True))

scene.build()
human_link_names = {link.idx: link.name for link in human.links}
print(human_link_names)
keys_pressed = set()

win = None
for attr in dir(scene.viewer):
    try:
        obj = getattr(scene.viewer, attr)
        if isinstance(obj, pyglet.window.Window):
            win = obj
            print(f"[IMU] Found pyglet window at scene.viewer.{attr}")
            break
    except Exception:
        continue

if win is None:
    all_wins = list(pyglet.app.windows)
    if all_wins:
        win = all_wins[0]
        print("[IMU] Using pyglet.app.windows fallback")
    else:
        raise RuntimeError("Could not find a pyglet window.")

win.push_handlers(
    on_key_press   = lambda sym, mod: keys_pressed.add(sym),
    on_key_release = lambda sym, mod: keys_pressed.discard(sym),
)

jnt_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint',      'wrist_2_joint',       'wrist_3_joint',
]
end_effector = ur5e.get_link('wrist_3_link')
probe_link   = ur5e.get_link('probe_link')
dofs_idx     = [ur5e.get_joint(name).dof_idx_local for name in jnt_names]

ur5e.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000]), dofs_idx)
ur5e.set_dofs_kv(np.array([450,  450,  350,  350,  200,  200 ]), dofs_idx)
ur5e.set_dofs_force_range(
    np.array([-150, -150, -150, -28, -28, -28]),
    np.array([ 150,  150,  150,  28,  28,  28]),
    dofs_idx,
)

home_joint_angles = np.array([
     0.0,
    -1.5708,
     1.5708,
    -1.5708,
    -1.5708,
     0.0,
])

print("Moving arm to home pose")
for _ in range(300):
    ur5e.control_dofs_position(home_joint_angles, dofs_idx_local=dofs_idx)
    scene.step()

actual_pos  = ur5e.get_link('wrist_3_link').get_pos()
actual_quat = ur5e.get_link('wrist_3_link').get_quat()
if hasattr(actual_pos, 'cpu'):
    actual_pos  = actual_pos.cpu().numpy()
    actual_quat = actual_quat.cpu().numpy()

ee_home_pos  = np.array(actual_pos,  dtype=float)
ee_home_quat = np.array(actual_quat, dtype=float)
ee_home_quat /= np.linalg.norm(ee_home_quat)

print(f"[INIT] Home pos {np.round(ee_home_pos,  3)}")
print(f"[INIT] Home quat {np.round(ee_home_quat, 4)}")
print("[INIT] Teleoperation ready")

print(f"[IMU] Waiting for first orientation sample on {IMU_TOPIC} ...")
_wait_start = time.time()
while True:
    _raw_quat, _got = imu_node.get_quat()
    if _got:
        break
    if time.time() - _wait_start > 10.0:
        print(f"[IMU] WARNING: no data received on {IMU_TOPIC} after 10s — "
              f"check the topic name, that the micro-ROS agent is running, "
              f"and QoS compatibility. Continuing with identity orientation.")
        break
    time.sleep(0.05)

imu_ref_quat_raw = _raw_quat.copy()
print(f"[IMU] Calibrated. Raw ref quat (w,x,y,z): {np.round(imu_ref_quat_raw, 4)}")

print(f"[IMU] Waiting for first position sample on {POS_TOPIC} ...")
_wait_start = time.time()
pos_ref_raw = np.array([0.0, 0.0, 0.0])
while True:
    _raw_pos, _got_pos = imu_node.get_pos()
    if _got_pos:
        pos_ref_raw = _raw_pos.copy()   # ESP32's filtered position at calibration = "home"
        break
    if time.time() - _wait_start > 10.0:
        print(f"[IMU] WARNING: no data received on {POS_TOPIC} after 10s — "
              f"check the ESP32 is publishing on this topic. Continuing with zero offset.")
        break
    time.sleep(0.05)
print(f"[IMU] Position calibrated. Ref pos (m): {np.round(pos_ref_raw, 4)}")

imu_pos      = np.array([1.5, 0.0, 1.0], dtype=float)
imu_quat     = np.array([0.5, -0.5, 0.5, -0.5], dtype=float)
imu_ref_pos  = imu_pos.copy()
imu_ref_quat = imu_quat.copy()

imu_quat_ee     = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
imu_ref_quat_ee = imu_quat_ee.copy()

ee_target_pos  = ee_home_pos.copy()
ee_target_quat = ee_home_quat.copy()

MOVE_STEP     = 0.005
ROT_STEP      = 0.02
POS_SCALE     = 1.0
MAX_TILT_RAD  = np.radians(90)
MAX_STEP_RAD  = 0.08
MAX_JUMP_RAD  = 10.3

POS_STALE_THRESHOLD_S = 0.3   # if ESP32 stops publishing position this long, freeze imu_pos

FORCE_PRINT_EVERY = 30
step_count = 0

last_good_qpos = None
current_qpos   = None

BODY_PART_IMAGE_FOLDERS = {
    'T8':     '/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/USG_data/T8/',
    'L5':     '/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/USG_data/L5/',
    'L3':     '/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/USG_data/L3',
    'T12':     '/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/USG_data/T12',
}
IMAGE_FRAME_RATE = 10


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def axis_angle_to_quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s])

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_to_euler_deg(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2 * (x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])

# R_CALIB is still needed here for ORIENTATION (imu_to_ee_target); it's no
# longer applied to acceleration in Python since that rotation now happens
# on the ESP32 before the Kalman filter runs.
R_CALIB = axis_angle_to_quat([1, 0, 0], np.pi / 2)

def frame_transform(delta, R):
    """Re-express a local-frame delta rotation in another frame's local axes."""
    return quat_mul(quat_mul(R, delta), quat_conjugate(R))


KEY_MAP = {
    # Translation keys removed -- position comes entirely from the ESP32's
    # Kalman+ZUPT-filtered /probe/position topic now.
}

def clamp_rotation(q, q_ref, max_angle):
    delta = quat_mul(quat_conjugate(q_ref), q)

    w = np.clip(delta[0], -1.0, 1.0)
    total_angle = 2 * np.arccos(abs(w))

    if total_angle > max_angle:
        axis = delta[1:]
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            half = max_angle / 2
            clamped_delta = np.array([
                np.cos(half),
                axis[0] * np.sin(half),
                axis[1] * np.sin(half),
                axis[2] * np.sin(half),
            ])
            q = quat_mul(q_ref, clamped_delta)
            q = q / np.linalg.norm(q)

    return q

def clamp_imu_rotation():
    global imu_quat, imu_quat_ee
    imu_quat    = clamp_rotation(imu_quat,    imu_ref_quat,    MAX_TILT_RAD)
    imu_quat_ee = clamp_rotation(imu_quat_ee, imu_ref_quat_ee, MAX_TILT_RAD)

def safe_ik(ee_pos, ee_quat):
    global last_good_qpos

    qpos = ur5e.inverse_kinematics(
        link=end_effector,
        pos=ee_pos,
        quat=ee_quat,
    )

    qpos = qpos.cpu().numpy()

    if not np.all(np.isfinite(qpos)):
        print("[WARN] IK returned NaN/Inf — holding last good pose")
        return last_good_qpos

    if last_good_qpos is not None:
        max_jump = np.max(np.abs(qpos - last_good_qpos))
        if max_jump > MAX_JUMP_RAD:
            print(f"[WARN] IK jump {max_jump:.2f} rad — holding last good pose")
            return last_good_qpos

    ur5e_limits = [
        (-2*np.pi, 2*np.pi),
        (-2*np.pi, 2*np.pi),
        (-np.pi,   np.pi  ),
        (-2*np.pi, 2*np.pi),
        (-2*np.pi, 2*np.pi),
        (-2*np.pi, 2*np.pi),
    ]
    for i, (lo, hi) in enumerate(ur5e_limits):
        if not (lo <= qpos[i] <= hi):
            print(f"[WARN] Joint {i} out of limits ({qpos[i]:.2f}) — holding")
            return last_good_qpos

    last_good_qpos = qpos.copy()
    return qpos

def smooth_ik(target_qpos):
    global current_qpos

    if current_qpos is None:
        current_qpos = target_qpos.copy()
        return current_qpos

    delta = np.clip(target_qpos - current_qpos, -MAX_STEP_RAD, MAX_STEP_RAD)
    current_qpos = current_qpos + delta
    return current_qpos

STALE_THRESHOLD_S = 0.15
_was_stale = False
_pos_was_stale = False

def update_imu():
    global imu_pos, imu_quat, imu_quat_ee, imu_ref_quat_raw, pos_ref_raw
    global _was_stale, _pos_was_stale

    # ── Translation: read the ESP32's already-filtered position directly.
    # No integration, no Kalman math here anymore -- just apply the same
    # "delta from calibration reference" pattern already used for
    # orientation, so re-centering (SPACE) works the same way for both.
    pos_age = imu_node.pos_data_age()
    if pos_age > POS_STALE_THRESHOLD_S and not _pos_was_stale:
        print(f"[IMU] Position STALE: no new sample for {pos_age:.2f}s — check ESP32 link")
        _pos_was_stale = True
    elif pos_age <= POS_STALE_THRESHOLD_S and _pos_was_stale:
        print(f"[IMU] Position recovered after {pos_age:.3f}s gap")
        _pos_was_stale = False

    if pos_age <= POS_STALE_THRESHOLD_S:
        raw_pos, got_pos = imu_node.get_pos()
        if got_pos:
            imu_pos[:] = imu_ref_pos + (raw_pos - pos_ref_raw) * POS_SCALE
    # else: position feed lost -> imu_pos holds its last value rather than
    # snapping somewhere wrong from stale data.

    age = imu_node.data_age()
    if age > STALE_THRESHOLD_S and not _was_stale:
        print(f"[IMU] STALE: no new sample for {age:.2f}s — check the micro-ROS link")
        _was_stale = True
    elif age <= STALE_THRESHOLD_S and _was_stale:
        print(f"[IMU] Recovered after {age:.3f}s gap")
        _was_stale = False

    raw_quat, got_data = imu_node.get_quat()
    if got_data:
        delta_raw = quat_mul(quat_conjugate(imu_ref_quat_raw), raw_quat)
        delta_raw /= np.linalg.norm(delta_raw)

        delta_raw[1] *= -1
        delta_raw[2] *= -1
        imu_quat = quat_mul(imu_ref_quat, delta_raw)
        imu_quat_ee = quat_mul(imu_ref_quat_ee, delta_raw)

    if pyglet_key.SPACE in keys_pressed:
        imu_quat[:]    = imu_ref_quat
        imu_quat_ee[:] = imu_ref_quat_ee
        if got_data:
            imu_ref_quat_raw = raw_quat.copy()
        # Re-zero translation reference to wherever the ESP32-filtered
        # position currently is, mirroring the orientation re-zero above.
        raw_pos, got_pos = imu_node.get_pos()
        if got_pos:
            pos_ref_raw = raw_pos.copy()

    imu_quat    /= np.linalg.norm(imu_quat)
    imu_quat_ee /= np.linalg.norm(imu_quat_ee)

DEBUG_CALIBRATION = True
_calib_print_counter = 0

def imu_to_ee_target():
    global ee_target_pos, ee_target_quat, _calib_print_counter

    delta_pos     = (imu_pos - imu_ref_pos) * POS_SCALE
    ee_target_pos = np.clip(
        ee_home_pos + delta_pos,
        [-0.1, -0.6, 0.6],
        [ 0.7,  0.6, 1.8],
    )

    delta_quat_imu_ee = quat_mul(quat_conjugate(imu_ref_quat_ee), imu_quat_ee)
    delta_quat_ee     = frame_transform(delta_quat_imu_ee, R_CALIB)

    ee_target_quat = quat_mul(ee_home_quat, delta_quat_ee)
    ee_target_quat /= np.linalg.norm(ee_target_quat)

    if DEBUG_CALIBRATION:
        _calib_print_counter += 1
        if _calib_print_counter % 30 == 0:
            imu_euler = np.round(quat_to_euler_deg(delta_quat_imu_ee), 1)
            ee_euler  = np.round(quat_to_euler_deg(delta_quat_ee), 1)
            print(f"[CALIB] IMU-local roll/pitch/yaw: {imu_euler}  ->  "
                  f"EE-local roll/pitch/yaw: {ee_euler}")
            print(f"[CALIB] imu_pos delta: {np.round(imu_pos - imu_ref_pos, 4)}")


try:
    while True:
        update_imu()
        clamp_imu_rotation()
        imu_to_ee_target()

        imu_entity.set_pos(imu_pos)
        imu_entity.set_quat(imu_quat)

        raw_qpos = safe_ik(ee_target_pos, ee_target_quat)

        if raw_qpos is None:
            scene.step()
            continue

        final_qpos = smooth_ik(raw_qpos)

        ur5e.control_dofs_position(final_qpos, dofs_idx_local=dofs_idx)
        scene.step()
        step_count += 1


except KeyboardInterrupt:
    print("\n[IMU] Simulation stopped.")
finally:
    imu_node.destroy_node()
    rclpy.shutdown()