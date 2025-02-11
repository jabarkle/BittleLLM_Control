#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from bittle_msgs.msg import AprilTag, Yolo, BittlePath, BittlePathJSON
import numpy as np
import json
import heapq  # For A* priority queue

class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
    
        qos_profile = QoSProfile(depth=10)
    
        # -- Subscriptions --
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile)
        self.create_subscription(AprilTag, '/april_tag/bittle_pose', self.bittlebot_callback, qos_profile)
        self.create_subscription(Yolo, '/yolo/goals', self.goal_callback, qos_profile)
    
        # -- Publishers --
        self.path_pub = self.create_publisher(BittlePath, '/bittlebot/path', qos_profile)
        self.json_pub = self.create_publisher(BittlePathJSON, '/bittlebot/path_json', qos_profile)
        # Publisher for RViz visualization (nav_msgs/Path)
        self.path_vis_pub = self.create_publisher(Path, '/bittlebot/path_visualization', qos_profile)
    
        # Internal state
        self.occupancy_grid = None
        self.bittlebot_position = None
        self.goal_position = None
        self.map_resolution = 0.0053  # Default resolution (meters per pixel)
        self.map_width = 320
        self.map_height = 240
    
        self.get_logger().info("PathPlanner node started!")
    
    def map_callback(self, msg: OccupancyGrid):
        """ Updates the occupancy grid. """
        self.occupancy_grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.get_logger().info("Occupancy grid updated.")
    
    def bittlebot_callback(self, msg: AprilTag):
        """ Convert the AprilTag world coordinates into grid indices for path planning. """
        self.get_logger().info(f"📡 Raw AprilTag Position (world): {msg.position}")
        # Convert world coordinates (meters) to grid indices.
        grid_x = int(msg.position[0] / self.map_resolution)
        grid_y = int(msg.position[1] / self.map_resolution)
        self.bittlebot_position = (grid_x, grid_y)
        self.bittlebot_position = (
            max(0, min(self.map_width - 1, self.bittlebot_position[0])),
            max(0, min(self.map_height - 1, self.bittlebot_position[1]))
        )
        self.get_logger().info(f"✅ Corrected BittleBot position (grid): {self.bittlebot_position}")
    
    def goal_callback(self, msg: Yolo):
        """ Updates the goal position. """
        if len(msg.xywh) < 2:
            self.get_logger().warn("Invalid goal data received.")
            return
    
        self.goal_position = (
            int((msg.xywh[0] - 0) / self.map_resolution),
            int((msg.xywh[1] - 0) / self.map_resolution)
        )
    
        self.get_logger().info(f"Goal position updated: {self.goal_position}")
    
        # Trigger path planning
        self.plan_path()
    
    def plan_path(self):
        """ Runs A* pathfinding and publishes the result. """
        if self.occupancy_grid is None or self.bittlebot_position is None or self.goal_position is None:
            self.get_logger().warn("Missing data for path planning.")
            return
    
        start = self.bittlebot_position
        goal = self.goal_position
    
        self.get_logger().info(f"🚀 Planning path from {start} to {goal}")
    
        # Check bounds
        if not (0 <= start[0] < self.map_width and 0 <= start[1] < self.map_height):
            self.get_logger().error(f"❌ Start position {start} is out of bounds!")
            return
    
        if not (0 <= goal[0] < self.map_width and 0 <= goal[1] < self.map_height):
            self.get_logger().error(f"❌ Goal position {goal} is out of bounds!")
            return
    
        # Check if goal is inside an obstacle
        if self.occupancy_grid[goal[1], goal[0]] != 0:
            self.get_logger().error(f"🛑 Goal position {goal} is blocked by an obstacle!")
            return
    
        # Attempt A* pathfinding
        path = self.a_star(start, goal)
    
        if path:
            self.publish_path(path)
        else:
            self.get_logger().warn("⚠️ A* failed to find a valid path!")
    
    def a_star(self, start, goal):
        """ Implements A* pathfinding algorithm. """
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
    
        while open_set:
            _, current = heapq.heappop(open_set)
    
            if current == goal:
                return self.reconstruct_path(came_from, current)
    
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                tentative_g_score = g_score[current] + 1  # Assuming uniform cost
    
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
        return None
    
    def get_neighbors(self, node):
        """ Returns valid neighboring cells (4-connected grid). """
        x, y = node
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(nx, ny) for nx, ny in neighbors if self.is_valid(nx, ny)]
    
    def is_valid(self, x, y):
        """ Checks if a grid cell is valid (not an obstacle). """
        return (0 <= x < self.map_width and 0 <= y < self.map_height and self.occupancy_grid[y, x] == 0)
    
    def heuristic(self, node, goal):
        """ Euclidean distance heuristic. """
        return np.linalg.norm(np.array(node) - np.array(goal))
    
    def reconstruct_path(self, came_from, current):
        """ Reconstructs the shortest path from goal to start. """
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
    
    def publish_path(self, path):
        """ Publishes path using BittlePath and JSON for LLM, and publishes nav_msgs/Path for RViz. """
        path_msg = BittlePath()
        json_msg = BittlePathJSON()
        nav_path = Path()
        nav_path.header.stamp = self.get_clock().now().to_msg()
        nav_path.header.frame_id = "map"  # Ensure this matches your RViz Fixed Frame
    
        json_data = {"path": []}
    
        for (x, y) in path:
            pose = PoseStamped()
            pose.pose.position.x = x * self.map_resolution
            pose.pose.position.y = y * self.map_resolution
            pose.pose.position.z = 0.0
    
            path_msg.waypoints.append(pose)
            nav_path.poses.append(pose)
            json_data["path"].append([pose.pose.position.x, pose.pose.position.y])
    
        self.path_pub.publish(path_msg)
        json_msg.json_data = json.dumps(json_data)
        self.json_pub.publish(json_msg)
        self.path_vis_pub.publish(nav_path)  # Publish for RViz visualization
    
        self.get_logger().info("Path published!")
    
def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
