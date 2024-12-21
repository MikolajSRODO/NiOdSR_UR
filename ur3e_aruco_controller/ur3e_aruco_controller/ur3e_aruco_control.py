#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import cv2

class UR3eArucoController(Node):
    def __init__(self):
        super().__init__('ur3e_aruco_controller')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
        self.params = cv2.aruco.DetectorParameters_create()
        self.last_marker_position = None  # Śledzenie ostatniej pozycji markera

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

            if ids is not None and 7 in ids:
                marker_y = int(corners[0][0][:, 1].mean())
                if marker_y < gray.shape[0] / 2:
                    if self.last_marker_position != 'top':
                        self.publish_trajectory([1.0, -1.0, 0.5, 0.0, 0.0, 0.0])
                        self.last_marker_position = 'top'
                else:
                    if self.last_marker_position != 'bottom':
                        self.publish_trajectory([-1.0, -1.0, 0.9, 0.0, 0.0, 0.0])
                        self.last_marker_position = 'bottom'
            else:
                self.last_marker_position = None
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def publish_trajectory(self, positions):
        trajectory = JointTrajectory()
        trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 2
        trajectory.points.append(point)
        self.publisher.publish(trajectory)

def main():
    rclpy.init()
    node = UR3eArucoController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
