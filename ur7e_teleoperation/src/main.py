import genesis as gs
import numpy as np
import pyglet
from pyglet.window import key as pyglet_key

gs.init(backend=gs.gpu)

scene = gs.Scene(
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
    )
)

imu_entity = scene.add_entity(
    gs.morphs.Box(size=(0.06, 0.04, 0.01), pos=(1.5, 0.0, 1.0), fixed=True)
)

scene.build()

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

print("[INIT] Moving arm to home pose — please wait...")
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

print(f"[INIT] Home pos  = {np.round(ee_home_pos,  3)}")
print(f"[INIT] Home quat = {np.round(ee_home_quat, 4)}")
print("[INIT] Teleoperation ready")

# ── IMU state ──────────────────────────────────────────────────────────
imu_pos      = np.array([1.5, 0.0, 1.0], dtype=float)
imu_quat     = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
imu_ref_pos  = imu_pos.copy()
imu_ref_quat = imu_quat.copy()

ee_target_pos  = ee_home_pos.copy()
ee_target_quat = ee_home_quat.copy()

MOVE_STEP     = 0.005
ROT_STEP      = 0.02
POS_SCALE     = 1.0
MAX_TILT_RAD  = np.radians(60)
SMOOTH        = 0.15
MAX_JUMP_RAD  = 0.3

last_good_qpos = None
current_qpos   = None

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

KEY_MAP = {
    pyglet_key.UP       : ('pos', 0,  MOVE_STEP),
    pyglet_key.DOWN     : ('pos', 0, -MOVE_STEP),
    pyglet_key.RIGHT    : ('pos', 1,  MOVE_STEP),
    pyglet_key.LEFT     : ('pos', 1, -MOVE_STEP),
    pyglet_key.E        : ('pos', 2,  MOVE_STEP),
    pyglet_key.Q        : ('pos', 2, -MOVE_STEP),
    pyglet_key.T        : ('rot', [0,1,0],  ROT_STEP),
    pyglet_key.G        : ('rot', [0,1,0], -ROT_STEP),
    pyglet_key.F        : ('rot', [0,0,1],  ROT_STEP),
    pyglet_key.H        : ('rot', [0,0,1], -ROT_STEP),
    pyglet_key.V        : ('rot', [1,0,0],  ROT_STEP),
    pyglet_key.B        : ('rot', [1,0,0], -ROT_STEP),
}

def clamp_imu_rotation():
    global imu_quat

    delta = quat_mul(quat_conjugate(imu_ref_quat), imu_quat)

    w = np.clip(delta[0], -1.0, 1.0)
    total_angle = 2 * np.arccos(abs(w))

    if total_angle > MAX_TILT_RAD:
        axis = delta[1:]
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis /= axis_norm
            half = MAX_TILT_RAD / 2
            clamped_delta = np.array([
                np.cos(half),
                axis[0] * np.sin(half),
                axis[1] * np.sin(half),
                axis[2] * np.sin(half),
            ])
            imu_quat = quat_mul(imu_ref_quat, clamped_delta)
            imu_quat /= np.linalg.norm(imu_quat)

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
    global imu_pos, imu_quat

    for sym, action in KEY_MAP.items():
        if sym not in keys_pressed:
            continue
        if action[0] == 'pos':
            _, axis, delta = action
            imu_pos[axis] += delta
        else:
            _, ax, angle = action
            imu_quat = quat_mul(imu_quat, axis_angle_to_quat(ax, angle))

    if pyglet_key.SPACE in keys_pressed:
        imu_pos[:]  = imu_ref_pos
        imu_quat[:] = imu_ref_quat

    imu_quat /= np.linalg.norm(imu_quat)

def imu_to_ee_target():
    global ee_target_pos, ee_target_quat

    delta_pos     = (imu_pos - imu_ref_pos) * POS_SCALE
    ee_target_pos = np.clip(
        ee_home_pos + delta_pos,
        [-0.1, -0.6, 0.6],
        [ 0.7,  0.6, 1.8],
    )

    delta_quat     = quat_mul(quat_conjugate(imu_ref_quat), imu_quat)
    ee_target_quat = quat_mul(ee_home_quat, delta_quat)
    ee_target_quat /= np.linalg.norm(ee_target_quat)

print("=" * 52)
print("  IMU Teleoperation — controls")
print("  +X / -X  : UP    / DOWN")
print("  +Y / -Y  : RIGHT / LEFT")
print("  +Z / -Z  : E / Q")
print("  Pitch    : T / G")
print("  Yaw      : F / H")
print("  Roll     : V / B")
print("  Reset    : SPACE")
print("  Quit     : Ctrl-C")
print("=" * 52)

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

except KeyboardInterrupt:
    print("\n[IMU] Simulation stopped.")