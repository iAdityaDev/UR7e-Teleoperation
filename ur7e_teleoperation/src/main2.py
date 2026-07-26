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

gs.init(backend=gs.gpu)

# ── Live IMU input (BNO085 over micro-ROS) ───────────────────────────────
# CONFIGURE these two to match your setup:
IMU_TOPIC = '/imu/data'   # <-- change to the actual topic your micro-ROS agent publishes on

class BNO085Subscriber(Node):
    """Subscribes to the live IMU topic and exposes the latest orientation
    quaternion in a thread-safe way, reordered to this script's (w, x, y, z)
    convention (sensor_msgs/Imu.orientation is x, y, z, w)."""

    def __init__(self, topic_name):
        super().__init__('genesis_teleop_imu_subscriber')
        self._lock = threading.Lock()
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._has_data = False
        # micro-ROS publishers are almost always BEST_EFFORT; this profile
        # matches that. If you get zero messages, check
        # `ros2 topic info <topic> --verbose` and adjust QoS here.
        self.create_subscription(ImuMsg, topic_name, self._callback, qos_profile_sensor_data)
        self.get_logger().info(f'Subscribed to {topic_name}, waiting for data...')

    def _callback(self, msg):
        if len(msg.orientation_covariance) and msg.orientation_covariance[0] == -1.0:
            return  # driver is reporting "orientation not available"
        q = msg.orientation
        with self._lock:
            self._quat = np.array([q.w, q.x, q.y, q.z])
            self._has_data = True

    def get_quat(self):
        with self._lock:
            return self._quat.copy(), self._has_data


# signal_handler_options=NO keeps rclpy from grabbing SIGINT, so Ctrl+C
# still raises KeyboardInterrupt in the main loop exactly as before.
rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
imu_node = BNO085Subscriber(IMU_TOPIC)
threading.Thread(target=rclpy.spin, args=(imu_node,), daemon=True).start()

scene = gs.Scene(
    profiling_options=gs.options.ProfilingOptions(
        show_FPS=False,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
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
    gs.morphs.Box(size=(0.06, 0.04, 0.01), pos=(1.5, 0.0, 1.0), fixed=True)
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

print(f"[IMU] Waiting for first sample on {IMU_TOPIC} ...")
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

imu_ref_quat_raw = _raw_quat.copy()   # current physical sensor pose = "home"
print(f"[IMU] Calibrated. Raw ref quat (w,x,y,z): {np.round(imu_ref_quat_raw, 4)}")

imu_pos      = np.array([1.5, 0.0, 1.0], dtype=float)
imu_quat     = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
imu_ref_pos  = imu_pos.copy()
imu_ref_quat = imu_quat.copy()

# EE-facing rotation accumulator (kept separate from imu_quat so the
# imu_entity gizmo's visual behavior stays completely unchanged; this one
# has pitch/yaw inverted per-axis before composition, see EE_AXIS_SIGN).
imu_quat_ee     = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
imu_ref_quat_ee = imu_quat_ee.copy()

ee_target_pos  = ee_home_pos.copy()
ee_target_quat = ee_home_quat.copy()

MOVE_STEP     = 0.005
ROT_STEP      = 0.02
POS_SCALE     = 1.0
MAX_TILT_RAD  = np.radians(90)
SMOOTH        = 0.15
MAX_JUMP_RAD  = 3.0

FORCE_PRINT_EVERY = 30
step_count = 0

last_good_qpos = None
current_qpos   = None

# ── Ultrasound image-per-body-part config ────────────────────────────────
# Add a new body part -> just add one line here pointing at its frames folder.
BODY_PART_IMAGE_FOLDERS = {
    'T8':     '/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/USG_data/T8/',
    'L5':     '/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/USG_data/L5/',
    # 'Pelvis': '/home/deviant/IIIT_intern/assets/ultrasound_images/pelvis/',
    'L3':     '/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/USG_data/L3',
    'T12':     '/home/deviant/IIIT_intern/src/ur7e_teleoperation/assets/USG_data/T12',
}
IMAGE_FRAME_RATE = 10   # frames per second, tweak later


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

# ── Frame calibration (IMU-local axes -> EE-local axes) ─────────────────
# Confirmed: a 90 deg rotation about Z maps IMU-local Y (pitch) onto
# EE-local X (pitch/roll axis correctly). Sign flipped from -90 to +90
# to correct the direction (was rotating opposite to the IMU's motion).
R_CALIB = axis_angle_to_quat([0, 0, 1], -np.pi / 2)

def frame_transform(delta, R):
    """Re-express a local-frame delta rotation in another frame's local axes."""
    return quat_mul(quat_mul(R, delta), quat_conjugate(R))

KEY_MAP = {
    pyglet_key.UP       : ('pos', 0,  MOVE_STEP),
    pyglet_key.DOWN     : ('pos', 0, -MOVE_STEP),
    pyglet_key.RIGHT    : ('pos', 1,  MOVE_STEP),
    pyglet_key.LEFT     : ('pos', 1, -MOVE_STEP),
    pyglet_key.E        : ('pos', 2,  MOVE_STEP),
    pyglet_key.Q        : ('pos', 2, -MOVE_STEP),
}
# Rotation used to come from Y/U/J/K/B/N here — it now comes live from the
# BNO085 instead (see update_imu()), so only translation keys remain.

def mirror_pitch_yaw(q):
    """Replaces the old per-key EE_AXIS_SIGN. Negating a quaternion's y,z
    components re-expresses the same delta rotation with roll (x) left
    alone and pitch/yaw (y,z) inverted — exactly what EE_AXIS_SIGN did per
    keypress, just applied once to the live sensor delta instead of once
    per key. Flip the signs below first if the arm tilts opposite to your
    wrist once you're testing on hardware."""
    w, x, y, z = q
    return np.array([w, x, -y, -z])

def clamp_rotation(q, q_ref, max_angle):
    """Clamp q's total rotation relative to q_ref to at most max_angle.
    Generic version of the original clamp body, reusable for both
    imu_quat (visual gizmo) and imu_quat_ee (EE-driving)."""
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
    """Clamp BOTH accumulators independently: imu_quat (drives the visual
    gizmo) and imu_quat_ee (drives the arm). Previously only imu_quat was
    clamped here, so the gizmo respected MAX_TILT_RAD but the arm — which
    reads from imu_quat_ee — did not."""
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

    current_qpos = current_qpos + SMOOTH * (target_qpos - current_qpos)
    return current_qpos

def update_imu():
    global imu_pos, imu_quat, imu_quat_ee, imu_ref_quat_raw

    # Position: keyboard only — the BNO085 gives orientation, not translation.
    for sym, action in KEY_MAP.items():
        if sym not in keys_pressed:
            continue
        _, axis, delta = action
        imu_pos[axis] += delta

    # Orientation: pulled live from the sensor every frame.
    raw_quat, got_data = imu_node.get_quat()
    if got_data:
        delta_raw = quat_mul(quat_conjugate(imu_ref_quat_raw), raw_quat)
        delta_raw /= np.linalg.norm(delta_raw)

        # Visual gizmo — raw sensor delta, unmirrored, drives imu_entity directly.
        imu_quat = quat_mul(imu_ref_quat, delta_raw)
        # EE-facing accumulator — pitch/yaw mirrored, same effect as the old EE_AXIS_SIGN.
        imu_quat_ee = quat_mul(imu_ref_quat_ee, mirror_pitch_yaw(delta_raw))

    if pyglet_key.SPACE in keys_pressed:
        imu_pos[:]     = imu_ref_pos
        imu_quat[:]    = imu_ref_quat
        imu_quat_ee[:] = imu_ref_quat_ee
        if got_data:
            imu_ref_quat_raw = raw_quat.copy()   # re-zero to the current physical pose

    imu_quat    /= np.linalg.norm(imu_quat)
    imu_quat_ee /= np.linalg.norm(imu_quat_ee)

def imu_to_ee_target():
    global ee_target_pos, ee_target_quat

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


try:
    while True:
        update_imu()
        # clamp_imu_rotation()
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


except KeyboardInterrupt:
    print("\n[IMU] Simulation stopped.")
finally:
    imu_node.destroy_node()
    rclpy.shutdown()