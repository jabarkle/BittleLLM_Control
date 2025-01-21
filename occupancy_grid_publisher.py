#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import OccupancyGrid, MapMetaData
from bittle_msgs.msg import Yolo

class OccupancyGridPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_grid_publisher')
        qos_profile = QoSProfile(depth=10)

        # Subscribe to YOLO bounding boxes
        self.sub_detections = self.create_subscription(
            Yolo,
            '/yolo_topic',
            self.detections_callback,
            qos_profile
        )

        # Publisher for the occupancy grid
        self.pub_grid = self.create_publisher(OccupancyGrid, '/map', qos_profile)

        # Define some map parameters
        # (Assumes the overhead camera sees a 640x480 area and we treat each pixel as 1 'cell')
        # Adjust as needed
        self.map_resolution = 0.01  # e.g. each cell = 1 cm in the real world
        self.map_width  = 320      # in cells
        self.map_height = 240       # in cells

        self.get_logger().info("OccupancyGridPublisher node started. Subscribing to /yolo_topic.")

    def detections_callback(self, msg: Yolo):
        """
        We get bounding boxes in pixel coordinates (xywh_list).
        We’ll mark those bounding boxes in the OccupancyGrid as 'occupied' = 100.
        """

        # 1) Create a blank occupancy array with all free = 0
        occupancy_data = [0] * (self.map_width * self.map_height)

        # 2) Convert the flattened XYWH list into bounding boxes
        #    XYWH = [ x, y, w, h,  x2, y2, w2, h2, ... ]
        #    Typically x,y is the center, but if YOLO is returning absolute coords
        #    you'd want to confirm if x,y is center or top-left. By default, YOLO
        #    'xywh' means center x,y, width, height. So we offset accordingly.
        xywh = msg.xywh_list
        for i in range(0, len(xywh), 4):
            center_x = xywh[i + 0]
            center_y = xywh[i + 1]
            width    = xywh[i + 2]
            height   = xywh[i + 3]

            # Convert from center-based to top-left and bottom-right
            x1 = int(center_x - (width  / 2))
            y1 = int(center_y - (height / 2))
            x2 = int(center_x + (width  / 2))
            y2 = int(center_y + (height / 2))

            # Ensure bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.map_width - 1,  x2)
            y2 = min(self.map_height - 1, y2)

            # 3) Mark these bounding-box cells as occupied (100)
            for row in range(y1, y2 + 1):
                for col in range(x1, x2 + 1):
                    idx = row * self.map_width + col
                    occupancy_data[idx] = 100

        # 4) Create the OccupancyGrid message
        grid_msg = OccupancyGrid()

        # Header
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'  # or 'camera' or anything you prefer

        # Map MetaData
        meta = MapMetaData()
        meta.resolution = self.map_resolution  # meters per cell
        meta.width      = self.map_width
        meta.height     = self.map_height
        # origin = bottom-left corner in world space. For overhead, 0,0 is fine.
        meta.origin.position.x = 0.0
        meta.origin.position.y = 0.0
        meta.origin.orientation.w = 1.0
        grid_msg.info = meta

        # 5) Occupancy data
        grid_msg.data = occupancy_data

        # 6) Publish the occupancy grid
        self.pub_grid.publish(grid_msg)
        self.get_logger().info("Published occupancy grid with {} bounding boxes.".format(len(xywh)//4))

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
