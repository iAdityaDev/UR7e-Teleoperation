#!/usr/bin/env python3
"""
multi_apriltag_position_node.py

Tracks a 9-tag AprilTag bracket (1 top + 8 octagonal side tags) mounted on
the ultrasound probe. Detects all visible tags each frame, transforms each
detection to the probe's control point using a per-tag fixed offset (from
your CAD measurements), and picks the highest-confidence result.

Publishes:
  - probe/position          (geometry_msgs/PointStamped)   absolute control-point XYZ, camera frame
  - probe/position_delta    (geometry_msgs/Vector3Stamped) delta from calibrated zero
  - probe/pose_vision        (geometry_msgs/PoseStamped)    control-point pose (position + orientation)
  - probe/tracking_status   (std_msgs/Bool)                True = fresh detection this frame,
                                                             False = no tag visible (consumer should freeze)

Requires:
    pip install pyrealsense2 pupil-apriltags opencv-python --break-system-packages
"""

import numpy as np
import cv2
from pupil_apriltags import Detector
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from geometry_msgs.msg import PointStamped, Vector3Stamped, PoseStamped
from std_msgs.msg import Bool


# ══════════════════════════════════════════════════════════════════════
# TAG OFFSET TABLE — measured from the physical octagonal bracket
#
# For each tag: fixed transform FROM the tag's own frame TO the probe's
# control point (tip), expressed in the tag's local frame.
#
#   position_offset : (x, y, z) in meters, tag-frame -> control point
#   orientation_offset_euler_deg : (roll, pitch, yaw) in degrees
#
# Convention:
#   Tag ID 0   -> top tag (flat top face)
#   Tag ID 1-8 -> side tags, clockwise from probe "front", 45 deg apart
#
# Measured values:
#   Apothem (center-axis to side-tag face)   = 35 mm = 0.035 m
#   Top tag height above control point       = 50 mm = 0.050 m
#   Side tags are 15 mm below the top face
#     -> side tag height above control point = 50 - 15 = 35 mm = 0.035 m
#   Tag size (both top and side)             = 23 mm = 0.023 m
#
# AprilTag local-frame convention assumed: X = right in the printed
# image, Y = down in the printed image, Z = out of the tag face.
# ASSUMPTION (verify once running -- see notes below the table):
#   - Each side tag's printed "up" (i.e. -Y direction) points toward the
#     top tag along the probe's long axis.
#   - Each side tag's printed "right" (+X) points clockwise around the
#     octagon, matching your id 1->8 clockwise convention.
# If the printed markers were rotated differently when you stuck them on,
# swap/negate the relevant axis below to match -- symptoms of a wrong
# assumption are described after the table.
# ══════════════════════════════════════════════════════════════════════

TAG_BRACKET_APOTHEM_M     = 0.035   # side-tag face to center axis
TOP_TAG_HEIGHT_M          = 0.050   # top tag face to control point
SIDE_TAG_HEIGHT_M         = 0.035   # side tag center to control point (vertically)
TAG_SIZE_M                = 0.023   # both top and side tags

def _side_tag_offset(tag_id):
    """tag_id 1..8, clockwise from front (tag_id=1) at 45 deg spacing."""
    angle_deg = (tag_id - 1) * 45.0

    # Position offset in tag-local frame:
    #   Z: radially inward toward the center axis -> negative Z
    #   Y: down toward the control point along the probe axis -> positive Y
    #   X: 0 (control point is directly "below" this tag along its own Y)
    pos = np.array([
        0.0,
        SIDE_TAG_HEIGHT_M,
        -TAG_BRACKET_APOTHEM_M,
    ])

    # Orientation offset: yaw about the vertical (probe) axis to bring
    # this tag's frame into a common probe-reference frame, so published
    # orientation is consistent regardless of which tag is currently seen.
    orientation_euler_deg = np.array([0.0, 0.0, angle_deg])
    return pos, orientation_euler_deg

TAG_OFFSETS = {
    0: (np.array([0.0, 0.0, -TOP_TAG_HEIGHT_M]), np.array([0.0, 0.0, 0.0])),  # top tag
}
for _tid in range(1, 9):
    TAG_OFFSETS[_tid] = _side_tag_offset(_tid)

# ══════════════════════════════════════════════════════════════════════
# HOW TO VERIFY THE ASSUMPTION ABOVE:
# Run with show_debug_window=True. Hold the probe still and note the
# published control-point XYZ while only the TOP tag is visible. Then
# tilt slightly so a SIDE tag takes over. The published XYZ should stay
# essentially the same (control point didn't physically move).
#   - If X/Y jump sideways when handoff happens -> the "X=0" assumption
#     is wrong (tags aren't mounted with printed-up = toward-top); you'll
#     need to add a per-tag X offset or fix physical mounting rotation.
#   - If Z (depth) jumps a lot -> apothem or height measurement is off,
#     re-measure.
#   - If the position is roughly right but ~90 deg "rotated" in how X/Y
#     respond to probe rotation -> swap the sign/axis in _side_tag_offset
#     (X and Y roles may be flipped depending on how you physically
#     oriented each printed square when sticking it on).
# ══════════════════════════════════════════════════════════════════════


def euler_deg_to_rotmat(euler_deg):
    """Intrinsic XYZ euler (degrees) -> 3x3 rotation matrix."""
    r, p, y = np.radians(euler_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotmat_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([x, y, z, w])


class MultiAprilTagPositionNode(Node):
    def __init__(self):
        super().__init__('multi_apriltag_position_node')

        # ---- Parameters ----
        self.declare_parameter('tag_family', 'tag25h9')
        self.declare_parameter('tag_size_m', TAG_SIZE_M)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('depth_patch_radius', 2)
        self.declare_parameter('publish_frame', 'camera_color_optical_frame')
        self.declare_parameter('z_only_mode', False)
        self.declare_parameter('invert_z', False)
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter('quad_decimate', 1.0)
        self.declare_parameter('nthreads', 3)
        self.declare_parameter('min_decision_margin', 15.0)  # discard low-confidence detections

        self.tag_family = self.get_parameter('tag_family').value
        self.tag_size = self.get_parameter('tag_size_m').value
        self.width = self.get_parameter('frame_width').value
        self.height = self.get_parameter('frame_height').value
        self.fps = self.get_parameter('fps').value
        self.patch_radius = self.get_parameter('depth_patch_radius').value
        self.publish_frame = self.get_parameter('publish_frame').value
        self.z_only_mode = self.get_parameter('z_only_mode').value
        self.invert_z = self.get_parameter('invert_z').value
        self.show_debug = self.get_parameter('show_debug_window').value
        self.quad_decimate = self.get_parameter('quad_decimate').value
        self.nthreads = self.get_parameter('nthreads').value
        self.min_decision_margin = self.get_parameter('min_decision_margin').value

        self.known_tag_ids = set(TAG_OFFSETS.keys())

        # Precompute rotation matrices for each tag's orientation offset
        self.tag_offset_rotmats = {
            tid: euler_deg_to_rotmat(off_euler)
            for tid, (off_pos, off_euler) in TAG_OFFSETS.items()
        }

        # ---- RealSense pipeline setup ----
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        rs_config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = self.pipeline.start(rs_config)
        self.align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.camera_params = (
            self.intrinsics.fx, self.intrinsics.fy,
            self.intrinsics.ppx, self.intrinsics.ppy
        )

        self.get_logger().info(
            f'D435i started: {self.width}x{self.height}@{self.fps}fps, '
            f'depth_scale={self.depth_scale:.6f} m/unit'
        )
        self.get_logger().info(f'Tracking {len(self.known_tag_ids)} known tag IDs: {sorted(self.known_tag_ids)}')

        # ---- AprilTag detector ----
        self.detector = Detector(
            families=self.tag_family,
            nthreads=self.nthreads,
            quad_decimate=self.quad_decimate,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        # ---- Calibration / zero-reference state ----
        self.zero_position = None
        self.last_position = None

        # ---- Publishers ----
        self.pos_pub = self.create_publisher(PointStamped, 'probe/position', 10)
        self.delta_pub = self.create_publisher(Vector3Stamped, 'probe/position_delta', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'probe/pose_vision', 10)
        self.status_pub = self.create_publisher(Bool, 'probe/tracking_status', 10)

        self.calib_srv = self.create_service(
            Trigger, 'probe/calibrate_zero', self.calibrate_zero_callback)

        self.timer = self.create_timer(1.0 / self.fps, self.frame_callback)

        self.get_logger().info(
            'Watching for any of the known tags. Waiting for first detection '
            'to auto-calibrate zero. Call /probe/calibrate_zero to re-zero.'
        )

    # ------------------------------------------------------------------
    def calibrate_zero_callback(self, request, response):
        if self.last_position is not None:
            self.zero_position = self.last_position.copy()
            response.success = True
            response.message = f'Zero calibrated at {self.zero_position.tolist()}'
        else:
            response.success = False
            response.message = 'No tag position available yet to calibrate.'
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------
    def frame_callback(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size
        )

        now = self.get_clock().now().to_msg()

        # ---- Filter to known tags with acceptable confidence ----
        candidates = [
            d for d in detections
            if d.tag_id in self.known_tag_ids and d.decision_margin >= self.min_decision_margin
        ]

        if not candidates:
            # No usable detection this frame -> publish tracking_status=False
            # and DO NOT publish a new position (consumer holds last value).
            status_msg = Bool()
            status_msg.data = False
            self.status_pub.publish(status_msg)
            if self.show_debug:
                self._show_debug(color_img, None, None, None, None)
            return

        # ---- Pick best-confidence detection ----
        best = max(candidates, key=lambda d: d.decision_margin)

        corners = best.corners
        center_px = best.center
        cx_px, cy_px = float(center_px[0]), float(center_px[1])

        depth_m = self._get_patch_depth(depth_img, cx_px, cy_px)
        if depth_m is None or depth_m <= 0.0:
            self.get_logger().warn('Invalid depth at tag center, skipping frame.', throttle_duration_sec=2.0)
            status_msg = Bool()
            status_msg.data = False
            self.status_pub.publish(status_msg)
            if self.show_debug:
                self._show_debug(color_img, corners, None, None, best.tag_id)
            return

        tag_center_3d = np.array(
            rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx_px, cy_px], depth_m),
            dtype=np.float64
        )

        # ---- Transform tag detection -> probe control point ----
        # Tag orientation in camera frame (from AprilTag's own pose_R).
        R_cam_tag = np.array(best.pose_R)  # camera <- tag rotation

        off_pos, off_euler = TAG_OFFSETS[best.tag_id]
        R_offset = self.tag_offset_rotmats[best.tag_id]  # tag-local rotation offset

        # Control point in camera frame:
        #   camera_frame_offset = R_cam_tag @ (tag-local offset position)
        #   control_point = tag_center_3d + camera_frame_offset
        control_point_3d = tag_center_3d + R_cam_tag @ off_pos

        # Control point orientation in camera frame:
        R_cam_control = R_cam_tag @ R_offset

        self.last_position = control_point_3d

        if self.zero_position is None:
            self.zero_position = control_point_3d.copy()
            self.get_logger().info(f'Auto-calibrated zero at {self.zero_position.tolist()}')

        # ---- Publish absolute position ----
        pt_msg = PointStamped()
        pt_msg.header.stamp = now
        pt_msg.header.frame_id = self.publish_frame
        pt_msg.point.x, pt_msg.point.y, pt_msg.point.z = control_point_3d.tolist()
        self.pos_pub.publish(pt_msg)

        # ---- Publish delta from zero ----
        delta = control_point_3d - self.zero_position
        z_delta = -delta[2] if self.invert_z else delta[2]

        delta_msg = Vector3Stamped()
        delta_msg.header.stamp = now
        delta_msg.header.frame_id = self.publish_frame
        if self.z_only_mode:
            delta_msg.vector.x = 0.0
            delta_msg.vector.y = 0.0
        else:
            delta_msg.vector.x = float(delta[0])
            delta_msg.vector.y = float(delta[1])
        delta_msg.vector.z = float(z_delta)
        self.delta_pub.publish(delta_msg)

        # ---- Publish pose ----
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.publish_frame
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = control_point_3d.tolist()
        quat = rotmat_to_quat(R_cam_control)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

        # ---- Tracking status: fresh detection this frame ----
        status_msg = Bool()
        status_msg.data = True
        self.status_pub.publish(status_msg)

        if self.show_debug:
            self._show_debug(color_img, corners, control_point_3d, delta, best.tag_id, best.decision_margin)

    # ------------------------------------------------------------------
    def _get_patch_depth(self, depth_img, cx_px, cy_px):
        h, w = depth_img.shape[:2]
        cx, cy = int(round(cx_px)), int(round(cy_px))
        r = self.patch_radius

        x0, x1 = max(cx - r, 0), min(cx + r + 1, w)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, h)

        patch = depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        median_raw = float(np.median(valid))
        return median_raw * self.depth_scale

    def _show_debug(self, color_img, corners, point_3d=None, delta=None, tag_id=None, decision_margin=None):
        disp = color_img.copy()
        if corners is not None:
            pts = corners.astype(int)
            cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
            center = pts.mean(axis=0).astype(int)
            cv2.circle(disp, tuple(center), 4, (0, 0, 255), -1)
            cv2.putText(disp, f'tag {tag_id}', tuple(center + np.array([8, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if point_3d is not None:
                txt = f'ctrl pt XYZ: {point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f} m'
                cv2.putText(disp, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if delta is not None:
                txt2 = f'dXYZ mm: {delta[0]*1000:.1f}, {delta[1]*1000:.1f}, {delta[2]*1000:.1f}'
                cv2.putText(disp, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            if decision_margin is not None:
                cv2.putText(disp, f'confidence: {decision_margin:.1f}', (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(disp, 'NO TAG VISIBLE - FROZEN', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Multi-Tag Probe Tracking', disp)
        cv2.waitKey(1)

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        if self.show_debug:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiAprilTagPositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()