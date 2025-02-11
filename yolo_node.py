#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from bittle_msgs.msg import Yolo  # Custom message

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to camera stream
        self.subscription = self.create_subscription(
            Image,
            '/camera/stream',
            self.listener_callback,
            10
        )

        # Load YOLO model
        self.model = YOLO('/home/jesse/ros2_ws/src/bittle_ros2/bittle_ros2/utils/best.pt')

        # Create ROS2 publishers
        self.obstacles_pub = self.create_publisher(Yolo, '/yolo/obstacles', 10)
        self.goals_pub = self.create_publisher(Yolo, '/yolo/goals', 10)
        self.annotated_image_pub = self.create_publisher(Image, '/yolo/detections', 10)

        # Define camera properties (Ensure they match your setup)
        self.image_width = 640  # Adjust if needed
        self.image_height = 480
        self.map_resolution = 0.0053  # 5.3 mm per pixel (adjust for real-world scaling)
        self.map_origin_x = 0.0  # Adjust if your map has an offset
        self.map_origin_y = 0.0

        self.get_logger().info("YOLO Node started!")

    def listener_callback(self, msg: Image):
        """ Process YOLO detections and publish obstacles & goals. """
        # Convert ROS2 Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Perform YOLO inference
        try:
            results = self.model.predict(frame)
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {str(e)}")
            return

        # Ensure valid detections
        if not results or len(results[0].boxes) == 0:
            return  # No detections

        # Extract detection data
        detections = results[0].boxes
        classes = detections.cls.cpu().tolist()  # Class IDs
        xywh = detections.xywh.cpu().tolist()   # Bounding boxes (pixels)
        xywhn = detections.xywhn.cpu().tolist() # Normalized bounding boxes

        # Initialize messages
        obstacles_msg, goals_msg = Yolo(), Yolo()
        found_goal = False  # Track if at least one goal is found

        # Process detections
        for i, class_id in enumerate(classes):
            class_id = int(class_id)
            if class_id not in [0, 1, 2]:  # Ensure valid classes (Goal = 0, Obstacle = 1, Goal Alternative = 2)
                continue

            bbox_xywh = xywh[i]
            bbox_xywhn = xywhn[i]

            # Convert pixel positions to world positions
            world_x = (bbox_xywh[0] * self.map_resolution) + self.map_origin_x
            world_y = (bbox_xywh[1] * self.map_resolution) + self.map_origin_y

            if class_id == 1:  # Obstacle
                obstacles_msg.class_ids.append(class_id)
                obstacles_msg.xywh.extend(map(float, bbox_xywh))
                obstacles_msg.xywhn.extend(map(float, bbox_xywhn))

            elif class_id in [0, 2]:  # Goal
                if len(bbox_xywh) < 2:
                    continue  # Skip invalid bounding boxes
                
                goals_msg.class_ids.append(class_id)
                goals_msg.xywh.extend([world_x, world_y, float(bbox_xywh[2]), float(bbox_xywh[3])])  # Convert to meters
                goals_msg.xywhn.extend(map(float, bbox_xywhn))
                found_goal = True

                self.get_logger().info(f"🎯 Goal detected at pixels: {bbox_xywh[:2]}")
                self.get_logger().info(f"🌍 Converted goal to world position: ({world_x}, {world_y})")

        # Publish goals if detected
        if found_goal:
            self.goals_pub.publish(goals_msg)

        # Publish obstacles if detected
        if obstacles_msg.class_ids:
            self.obstacles_pub.publish(obstacles_msg)

        # Publish annotated image (for debugging)
        annotated_msg = self.bridge.cv2_to_imgmsg(results[0].plot(), encoding='bgr8')
        self.annotated_image_pub.publish(annotated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
