#!/usr/bin/env python3
"""
d435i_apriltag_position_node.py

Standalone ROS2 node using pyrealsense2 DIRECTLY (no realsense2_camera driver
needed) to stream color+depth from a D435i, detect an AprilTag on the probe,
estimate its 3D position (X, Y, Z) via depth deprojection, and publish:

  - probe/position          (geometry_msgs/PointStamped) -> absolute XYZ, meters, camera frame
  - probe/position_delta    (geometry_msgs/Vector3Stamped) -> delta from calibrated zero, meters
  - probe/pose_vision        (geometry_msgs/PoseStamped) -> full 6DOF (tag orientation)

Z-axis is highlighted separately since that's what's driving your arm's
z-motion: probe/position_delta.vector.z is the signed depth-axis displacement
from the reference pose, ready to feed into your Genesis IK step as a
z-offset command.

Calibration: on startup (or on service call), the node captures the current
tag position as the "zero" reference. All deltas are computed relative to it.

Requires:
    pip install pyrealsense2 pupil-apriltags opencv-python --break-system-packages

Note: pupil-apriltags wraps the reference AprilTag C library (better pose
accuracy / corner refinement / false-positive rejection than cv2.aruco).
"""

import numpy as np
import cv2
from pupil_apriltags import Detector
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from geometry_msgs.msg import PointStamped, Vector3Stamped, PoseStamped


class D435iAprilTagPositionNode(Node):
    def __init__(self):
        super().__init__('d435i_apriltag_position_node')

        # ---- Parameters ----
        self.declare_parameter('tag_id', 0)
        self.declare_parameter('tag_family', 'tag36h11')  # most common/robust family
        self.declare_parameter('tag_size_m', 0.03)          # physical black-square side length, meters
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('depth_patch_radius', 2)
        self.declare_parameter('publish_frame', 'camera_color_optical_frame')
        self.declare_parameter('z_only_mode', False)
        self.declare_parameter('invert_z', False)
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter('quad_decimate', 1.0)   # >1.0 speeds up detection at cost of range/accuracy
        self.declare_parameter('nthreads', 2)

        self.tag_id = self.get_parameter('tag_id').value
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

        self.get_logger().info(
            f'D435i started: {self.width}x{self.height}@{self.fps}fps, '
            f'depth_scale={self.depth_scale:.6f} m/unit'
        )
        self.get_logger().info(
            f'Intrinsics: fx={self.intrinsics.fx:.2f} fy={self.intrinsics.fy:.2f} '
            f'ppx={self.intrinsics.ppx:.2f} ppy={self.intrinsics.ppy:.2f}'
        )

        # ---- AprilTag detector setup ----
        # pupil-apriltags wants camera params as (fx, fy, cx, cy) to do its own
        # internal pose estimation (pose_R, pose_t) in addition to raw corners.
        self.camera_params = (
            self.intrinsics.fx, self.intrinsics.fy,
            self.intrinsics.ppx, self.intrinsics.ppy
        )

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

        # ---- Service to (re)calibrate zero on demand ----
        self.calib_srv = self.create_service(
            Trigger, 'probe/calibrate_zero', self.calibrate_zero_callback)

        # ---- Timer-driven frame loop ----
        self.timer = self.create_timer(1.0 / self.fps, self.frame_callback)

        self.get_logger().info(
            f'Watching for tag_id={self.tag_id} family={self.tag_family}. '
            'Waiting for first detection to auto-calibrate zero. '
            'Call /probe/calibrate_zero to re-zero at any time.'
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

        target = None
        for det in detections:
            if det.tag_id == self.tag_id:
                target = det
                break

        if target is None:
            if self.show_debug:
                self._show_debug(color_img, None)
            return

        corners = target.corners  # shape (4,2), float
        center_px = target.center  # (cx, cy) float, already computed by the library
        cx_px, cy_px = float(center_px[0]), float(center_px[1])

        # ---- Depth-based Z (more reliable than the library's own pose_t depth
        # for close-range / noisy-corner scenarios, since it's a direct sensor
        # reading rather than a homography-based estimate) ----
        depth_m = self._get_patch_depth(depth_img, cx_px, cy_px)
        if depth_m is None or depth_m <= 0.0:
            self.get_logger().warn('Invalid depth at tag center, skipping frame.', throttle_duration_sec=2.0)
            if self.show_debug:
                self._show_debug(color_img, corners)
            return

        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx_px, cy_px], depth_m)
        point_3d = np.array(point_3d, dtype=np.float64)

        self.last_position = point_3d

        if self.zero_position is None:
            self.zero_position = point_3d.copy()
            self.get_logger().info(f'Auto-calibrated zero at {self.zero_position.tolist()}')

        # ---- Publish absolute position ----
        pt_msg = PointStamped()
        pt_msg.header.stamp = now
        pt_msg.header.frame_id = self.publish_frame
        pt_msg.point.x, pt_msg.point.y, pt_msg.point.z = point_3d.tolist()
        self.pos_pub.publish(pt_msg)

        # ---- Publish delta from zero (drives arm motion) ----
        delta = point_3d - self.zero_position
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

        # ---- Full pose: position from depth (robust), orientation from
        # AprilTag's own pose_R (library computes this via homography +
        # refinement, generally solid for orientation) ----
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.publish_frame
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = point_3d.tolist()

        if target.pose_R is not None:
            quat = self._rotmat_to_quat(np.array(target.pose_R))
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

        if self.show_debug:
            self._show_debug(color_img, corners, point_3d, delta, target.decision_margin)

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

    def _show_debug(self, color_img, corners, point_3d=None, delta=None, decision_margin=None):
        disp = color_img.copy()
        if corners is not None:
            pts = corners.astype(int)
            cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
            center = pts.mean(axis=0).astype(int)
            cv2.circle(disp, tuple(center), 4, (0, 0, 255), -1)
            if point_3d is not None:
                txt = f'XYZ: {point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f} m'
                cv2.putText(disp, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if delta is not None:
                txt2 = f'dZ: {delta[2]*1000:.1f} mm'
                cv2.putText(disp, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            if decision_margin is not None:
                txt3 = f'confidence: {decision_margin:.1f}'
                cv2.putText(disp, txt3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(disp, 'No tag detected', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('D435i AprilTag Tracking', disp)
        cv2.waitKey(1)

    @staticmethod
    def _rotmat_to_quat(R):
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
        return np.array([x, y, z, w], dtype=np.float64)

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
    node = D435iAprilTagPositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()