#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from bittle_msgs.msg import Yolo, AprilTag  # Importing both Yolo and AprilTag messages
import numpy as np

class OccupancyGridPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_grid_publisher')
        qos_profile = QoSProfile(depth=10)

        # -- Subscribe to YOLO obstacle detections --
        self.sub_yolo = self.create_subscription(
            Yolo,
            '/yolo/obstacles',  # Corrected topic
            self.yolo_callback,
            qos_profile
        )

        # -- Subscribe to AprilTag BittleBot Position --
        self.sub_apriltag = self.create_subscription(
            AprilTag,
            '/april_tag/bittle_pose',  # BittleBot position
            self.apriltag_callback,
            qos_profile
        )

        # -- Publisher for the occupancy grid --
        self.pub_grid = self.create_publisher(OccupancyGrid, '/map', qos_profile)
        self.pub_visualization = self.create_publisher(OccupancyGrid, '/visualization', qos_profile)  # RViz visualization

        # -- Define map parameters --
        self.map_width = 320   # Adjust based on your real-world scaling
        self.map_height = 240
        self.map_resolution = 0.0053  # meters per pixel (adjust as needed)

        # -- Info message for the map --
        self.map_info = MapMetaData()
        self.map_info.resolution = self.map_resolution
        self.map_info.width = self.map_width
        self.map_info.height = self.map_height
        self.map_info.origin = Pose()
        self.map_info.origin.position.x = 0.0
        self.map_info.origin.position.y = 0.0
        self.map_info.origin.position.z = 0.0
        self.map_info.origin.orientation.w = 1.0

        # Store BittleBot's position (initialize as None)
        self.bittlebot_position = None

        self.get_logger().info("OccupancyGridPublisher node has been started.")

    def apriltag_callback(self, apriltag_msg: AprilTag):
        """
        Updates the stored BittleBot position from the AprilTag node.
        """
        self.bittlebot_position = [apriltag_msg.position[0], apriltag_msg.position[1]]  # (x, y) position
        self.get_logger().info(f"Updated BittleBot position: {self.bittlebot_position}")

    def yolo_callback(self, yolo_msg: Yolo):
        """
        Updates the occupancy grid with obstacle data from YOLO while ignoring BittleBot’s position.
        """
        # 1. Create an empty occupancy grid array, initialized as all free space (0)
        occupancy_data = [0 for _ in range(self.map_width * self.map_height)]

        # Compute the inflation buffer in cells. 3 inches = 0.0762 m.
        buffer_cells = int(round(0.0762 / self.map_resolution))

        # 2. Extract obstacle bounding boxes from YOLO
        xywh_list = yolo_msg.xywh

        # 3. Iterate through bounding boxes and mark them as occupied
        for i in range(0, len(xywh_list), 4):
            x_center = xywh_list[i + 0]
            y_center = xywh_list[i + 1]
            w = xywh_list[i + 2]
            h = xywh_list[i + 3]

            # Compute min and max corners (in grid/pixel coordinates)
            x_min = int(x_center - w / 2.0)
            x_max = int(x_center + w / 2.0)
            y_min = int(y_center - h / 2.0)
            y_max = int(y_center + h / 2.0)

            # Inflate the bounding box by the buffer (and then clamp to map boundaries)
            x_min = max(0, x_min - buffer_cells)
            x_max = min(self.map_width - 1, x_max + buffer_cells)
            y_min = max(0, y_min - buffer_cells)
            y_max = min(self.map_height - 1, y_max + buffer_cells)

            # Fill the bounding box region in occupancy_data with 100 (occupied)
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if self.is_bittlebot_cell(x, y):  # Check if this cell contains the BittleBot
                        continue  # Skip marking the BittleBot's position as an obstacle

                    idx = y * self.map_width + x
                    occupancy_data[idx] = 100  # Mark as occupied

        # 4. Construct the OccupancyGrid message
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"
        grid_msg.info = self.map_info
        grid_msg.data = occupancy_data

        # 5. Publish the occupancy grid
        self.pub_grid.publish(grid_msg)
        self.pub_visualization.publish(grid_msg)  # Publish to RViz visualization topic

        self.get_logger().info("Published updated occupancy grid.")

    def is_bittlebot_cell(self, x, y):
        """
        Checks if the given (x, y) map cell corresponds to the BittleBot’s location.
        """
        if self.bittlebot_position is None:
            return False  # No valid BittleBot position available

        bittlebot_x, bittlebot_y = self.bittlebot_position

        # Convert real-world position to grid indices
        grid_x = int(bittlebot_x / self.map_resolution)
        grid_y = int(bittlebot_y / self.map_resolution)

        # Define a small area around the BittleBot to keep free
        safe_radius = 5  # Adjust as needed

        if abs(x - grid_x) <= safe_radius and abs(y - grid_y) <= safe_radius:
            return True  # This cell contains the BittleBot, so it should not be marked as an obstacle

        return False

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
