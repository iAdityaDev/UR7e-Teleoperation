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

gs.init(backend=gs.gpu)

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
R_CALIB = axis_angle_to_quat([0, 0, 1], np.pi / 2)

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

    delta_quat_imu = quat_mul(quat_conjugate(imu_ref_quat), imu_quat)
    delta_quat_ee  = frame_transform(delta_quat_imu, R_CALIB)

    ee_target_quat = quat_mul(ee_home_quat, delta_quat_ee)
    ee_target_quat /= np.linalg.norm(ee_target_quat)

def get_probe_contact_force():
    """Return the 3D contact force vector on probe_link from contact with
    the human, or None if there's no probe contact right now."""
    contacts = ur5e.get_contacts(with_entity=human)
    n_contacts = len(contacts['position'])

    if n_contacts == 0:
        return None

    link_a = contacts['link_a']
    if hasattr(link_a, 'cpu'):
        link_a = link_a.cpu().numpy()

    link_b = contacts['link_b']
    if hasattr(link_b, 'cpu'):
        link_b = link_b.cpu().numpy()

    forces_a = contacts['force_a']
    if hasattr(forces_a, 'cpu'):
        forces_a = forces_a.cpu().numpy()

    forces_b = contacts['force_b']
    if hasattr(forces_b, 'cpu'):
        forces_b = forces_b.cpu().numpy()

    probe_mask = (link_a == probe_link.idx) | (link_b == probe_link.idx)
    if not np.any(probe_mask):
        return None

    a_hits = (link_a[probe_mask] == probe_link.idx)
    rows_force = np.where(
        a_hits[:, None], forces_a[probe_mask], forces_b[probe_mask]
    )
    return np.sum(rows_force, axis=0)

def print_contact_forces():
    """Print contact force specifically on probe_link (not the whole arm)."""
    force = get_probe_contact_force()
    if force is None:
        print("No probe contact")
    else:
        print(f"probe contact force = {np.round(force, 3)} N | "
              f"mag = {np.linalg.norm(force):.3f} N")

def get_probe_contact_body_part():
    """Return a list of (body_part_name, force_magnitude) tuples for every
    human link currently in contact with the probe, sorted by force
    magnitude descending. Returns None if there's no probe contact."""
    contacts = ur5e.get_contacts(with_entity=human)
    n_contacts = len(contacts['position'])

    if n_contacts == 0:
        return None

    link_a = contacts['link_a']
    if hasattr(link_a, 'cpu'):
        link_a = link_a.cpu().numpy()

    link_b = contacts['link_b']
    if hasattr(link_b, 'cpu'):
        link_b = link_b.cpu().numpy()

    forces_a = contacts['force_a']
    if hasattr(forces_a, 'cpu'):
        forces_a = forces_a.cpu().numpy()

    forces_b = contacts['force_b']
    if hasattr(forces_b, 'cpu'):
        forces_b = forces_b.cpu().numpy()

    probe_mask = (link_a == probe_link.idx) | (link_b == probe_link.idx)
    if not np.any(probe_mask):
        return None

    # Figure out which side is the probe vs the human for each contact row
    a_is_probe = (link_a[probe_mask] == probe_link.idx)

    # human-side link index and the force exerted on the human at that link
    human_link_idx = np.where(a_is_probe, link_b[probe_mask], link_a[probe_mask])
    human_force    = np.where(
        a_is_probe[:, None], forces_b[probe_mask], forces_a[probe_mask]
    )

    # Aggregate force magnitude per unique human link (a link can appear
    # in multiple contact points, e.g. probe tip touching a curved surface)
    part_forces = {}
    for idx, f in zip(human_link_idx, human_force):
        name = human_link_names.get(int(idx), f"unknown_link_{int(idx)}")
        part_forces[name] = part_forces.get(name, 0.0) + np.linalg.norm(f)

    if not part_forces:
        return None

    return sorted(part_forces.items(), key=lambda kv: kv[1], reverse=True)


def print_probe_body_part():
    """Print which body part(s) the probe is currently touching."""
    parts = get_probe_contact_body_part()
    if parts is None:
        print("Probe not touching any body part")
    else:
        primary_name, primary_force = parts[0]
        print(f"Probe touching: {primary_name}")
        if len(parts) > 1:
            others = ", ".join(f"{n} ({f:.3f} N)" for n, f in parts[1:])
            print(f"  [also: {others}]")
        else:
            print()


# ── Ultrasound image display ─────────────────────────────────────────────
# Advances to the next frame whenever the probe's position or orientation
# changes noticeably while touching a mapped body part.
POS_CHANGE_THRESH = 0.002   # meters
ROT_CHANGE_THRESH = 0.02    # radians

_frame_idx    = 0
_active_part  = None
_last_pos     = None
_last_quat    = None

def get_current_ultrasound_image(ee_pos, ee_quat):
    """Return the ultrasound frame image for the current probe contact.
    The frame advances only when ee_pos/ee_quat changes enough since the
    last check; otherwise the same frame keeps showing. Returns None if
    the probe isn't touching a body part that has images mapped."""
    global _frame_idx, _active_part, _last_pos, _last_quat

    parts = get_probe_contact_body_part()
    if not parts:
        _active_part = None
        return None

    part_name = parts[0][0]
    folder = BODY_PART_IMAGE_FOLDERS.get(part_name)
    if folder is None:
        _active_part = None
        return None

    frames = sorted(glob.glob(os.path.join(folder, '*.png')))
    if not frames:
        return None

    # reset playback when we start touching a new/different part
    if part_name != _active_part:
        _active_part = part_name
        _frame_idx   = 0
        _last_pos    = np.array(ee_pos, dtype=float)
        _last_quat   = np.array(ee_quat, dtype=float)
        return mpimg.imread(frames[_frame_idx])

    pos_delta = np.linalg.norm(np.array(ee_pos) - _last_pos)
    quat_dot  = np.clip(np.abs(np.dot(ee_quat, _last_quat)), -1.0, 1.0)
    rot_delta = 2 * np.arccos(quat_dot)

    if pos_delta >= POS_CHANGE_THRESH or rot_delta >= ROT_CHANGE_THRESH:
        _frame_idx = (_frame_idx + 1) % len(frames)
        _last_pos  = np.array(ee_pos, dtype=float)
        _last_quat = np.array(ee_quat, dtype=float)

    return mpimg.imread(frames[_frame_idx])


# ── Live plot: EEF orientation + probe contact force + ultrasound view ──
HISTORY_LEN     = 300   # rolling window length (samples)
PLOT_UPDATE_EVERY = 5   # redraw every N sim steps

t_hist       = deque(maxlen=HISTORY_LEN)
roll_hist    = deque(maxlen=HISTORY_LEN)
pitch_hist   = deque(maxlen=HISTORY_LEN)
yaw_hist     = deque(maxlen=HISTORY_LEN)
fmag_hist    = deque(maxlen=HISTORY_LEN)

plt.ion()
fig, (ax_orient, ax_force, ax_image) = plt.subplots(
    3, 1, figsize=(8, 14),
    gridspec_kw={'height_ratios': [1, 1, 2.5]}   # USG panel gets more vertical space
)
fig.canvas.manager.set_window_title("EEF Orientation, Contact Force & Ultrasound View")

line_roll,  = ax_orient.plot([], [], label="roll",  color="tab:red")
line_pitch, = ax_orient.plot([], [], label="pitch", color="tab:green")
line_yaw,   = ax_orient.plot([], [], label="yaw",   color="tab:blue")
ax_orient.set_ylabel("degrees")
ax_orient.set_title("End-Effector Orientation")
ax_orient.set_ylim(-190, 190)
ax_orient.legend(loc="upper right")
ax_orient.grid(True, alpha=0.3)

line_fmag, = ax_force.plot([], [], label="Net Force |F|", color="black", linewidth=2)
ax_force.set_ylabel("N")
ax_force.set_xlabel("sim step")
ax_force.set_title("Probe Net Contact Force")
ax_force.legend(loc="upper right")
ax_force.grid(True, alpha=0.3)

# Ultrasound image panel — blank until the probe touches a mapped body part
ax_image.set_title("Ultrasound View")
ax_image.axis('off')
image_artist = ax_image.imshow(np.zeros((10, 10, 3)))
image_artist.set_visible(False)

fig.tight_layout()

def update_live_plot(step, ee_quat, force):
    t_hist.append(step)

    roll, pitch, yaw = quat_to_euler_deg(ee_quat)
    roll_hist.append(roll)
    pitch_hist.append(pitch)
    yaw_hist.append(yaw)

    if force is None:
        fmag_hist.append(0.0)
    else:
        fmag_hist.append(np.linalg.norm(force))

    line_roll.set_data(t_hist, roll_hist)
    line_pitch.set_data(t_hist, pitch_hist)
    line_yaw.set_data(t_hist, yaw_hist)

    line_fmag.set_data(t_hist, fmag_hist)

    ax_orient.set_xlim(t_hist[0], t_hist[-1] + 1)
    ax_force.set_xlim(t_hist[0], t_hist[-1] + 1)

    f_max = max(10.0, max(fmag_hist))
    ax_force.set_ylim(0, f_max * 1.1)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

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

        step_count += 1
        if step_count % FORCE_PRINT_EVERY == 0:
            print_contact_forces()
            print_probe_body_part()

        if step_count % PLOT_UPDATE_EVERY == 0:
            ee_pos_now  = end_effector.get_pos()
            ee_quat_now = end_effector.get_quat()
            if hasattr(ee_pos_now, 'cpu'):
                ee_pos_now = ee_pos_now.cpu().numpy()
            if hasattr(ee_quat_now, 'cpu'):
                ee_quat_now = ee_quat_now.cpu().numpy()
            probe_force = get_probe_contact_force()
            update_live_plot(step_count, ee_quat_now, probe_force)

            img = get_current_ultrasound_image(ee_pos_now, ee_quat_now)
            if img is None:
                image_artist.set_visible(False)
            else:
                image_artist.set_data(img)
                image_artist.set_visible(True)

except KeyboardInterrupt:
    print("\n[IMU] Simulation stopped.")