#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import apriltag
import numpy as np
from bittle_msgs.msg import AprilTag

BITTLEBOT_TAG_ID = 1  # Set to the actual AprilTag ID of the BittleBot

class AprilTagNode(Node):
    def __init__(self):
        super().__init__('apriltag_node')
        
        # Subscribe to the camera stream
        self.subscription = self.create_subscription(
            Image,
            '/camera/stream',
            self.listener_callback,
            10
        )
        
        # Publisher for BittleBot’s position
        self.detection_publisher = self.create_publisher(AprilTag, '/april_tag/bittle_pose', 10)
        
        # Publisher for annotated image visualization
        self.image_publisher = self.create_publisher(Image, '/apriltag_detections', 10)
        
        # CvBridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        # Define the resolution (m per pixel) that is consistent with YOLO and occupancy grid
        self.map_resolution = 0.0053  # 5.3 mm per pixel
        
        # (We ignore any additional offsets so that the image (0,0) corresponds to world (0,0))
        
        # Create the AprilTag detector (done once)
        self.detector = apriltag.Detector()
        
        self.get_logger().info("AprilTagNode started!")

    def listener_callback(self, msg):
        """ Process the camera frame, detect AprilTags, and publish the robot’s position. """
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray_frame)
        
        for r in results:
            cX, cY = int(r.center[0]), int(r.center[1])
            if r.tag_id == BITTLEBOT_TAG_ID:
                self.get_logger().info(f"BittleBot AprilTag detected at ({cX}, {cY})")
                
                # Instead of using solvePnP, we use the detected image center directly.
                # Compute world coordinates as (pixel * resolution)
                world_x = cX * self.map_resolution
                world_y = cY * self.map_resolution
                self.get_logger().info(f"✅ Updated BittleBot position: Image ({cX}, {cY}) → World ({world_x:.3f}, {world_y:.3f})")
                
                april_tag_message = AprilTag()
                april_tag_message.tag_id = BITTLEBOT_TAG_ID
                # Publish the world coordinates (in meters)
                april_tag_message.position = [float(world_x), float(world_y), 0.0]
                self.detection_publisher.publish(april_tag_message)

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
