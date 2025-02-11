#!/usr/bin/env python3
"""
LLM_Reasoning Node

This node aggregates data from:
  - /april_tag/bittle_pose (robot’s current position)
  - /yolo/goals (goal location)
  - /yolo/obstacles (detected obstacles)
  - /map (occupancy grid)
  - /bittlebot/path_json (precomputed path)

It then compiles a JSON structure with the following format:

{
  "robot_position": [x, y, theta],
  "goal_position": [gx, gy],
  "obstacles": [[ox1, oy1], [ox2, oy2], ...],
  "occupancy_grid": [[0, 100, ...], ...],
  "planned_path": [[p1x, p1y], [p2x, p2y], ...]
}

This JSON is sent as the “user” message to GPT‑4 (via the OpenAI API), and the response
is expected to be a JSON command (e.g., {"command": "move_forward"}). The command is then published
to the “/bittle_cmd” topic.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
import json
import openai

# Import custom messages from your package (adjust the import path as needed)
from bittle_msgs.msg import AprilTag, Yolo, BittlePathJSON

class LLMReasoning(Node):
    def __init__(self):
        super().__init__('llm_reasoning')
        
        # --- Subscriptions ---
        self.create_subscription(
            AprilTag,
            '/april_tag/bittle_pose',
            self.apriltag_callback,
            10
        )
        self.create_subscription(
            Yolo,
            '/yolo/goals',
            self.goal_callback,
            10
        )
        self.create_subscription(
            Yolo,
            '/yolo/obstacles',
            self.obstacles_callback,
            10
        )
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.occupancy_grid_callback,
            10
        )
        self.create_subscription(
            BittlePathJSON,
            '/bittlebot/path_json',
            self.path_json_callback,
            10
        )
        
        # --- Publisher to bittle_cmd (for sending high-level commands) ---
        self.cmd_pub = self.create_publisher(String, '/bittle_cmd', 10)
        
        # --- Data storage variables ---
        self.robot_position = None     # Expected format: [x, y, theta]
        self.goal_position = None      # Expected format: [gx, gy]
        self.obstacles = []            # List of obstacle centers: [[ox, oy], ...]
        self.occupancy_grid = None     # 2D list (list of lists)
        self.planned_path = []         # List of waypoints: [[p1x, p1y], [p2x, p2y], ...]
        
        # --- Timer: periodically aggregate data and run reasoning ---
        self.timer_period = 5.0  # seconds (adjust as needed)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # --- Initialize OpenAI (GPT-4) API key ---
        openai.api_key = "YOUR_API_KEY"  # Replace with your actual API key
        
        self.get_logger().info("LLM_Reasoning node started.")

    # Callback: Update the robot’s position from the AprilTag node.
    def apriltag_callback(self, msg: AprilTag):
        # The AprilTag message provides [x, y, z]; we use x and y and set theta to 0 (if no orientation is provided)
        self.robot_position = [msg.position[0], msg.position[1], 0.0]
        self.get_logger().info(f"Robot position updated: {self.robot_position}")

    # Callback: Update the goal position from the YOLO goal detection.
    def goal_callback(self, msg: Yolo):
        # In the YOLO node for goals, msg.xywh is extended with [world_x, world_y, width, height]
        if len(msg.xywh) >= 2:
            self.goal_position = [msg.xywh[0], msg.xywh[1]]
            self.get_logger().info(f"Goal position updated: {self.goal_position}")
        else:
            self.get_logger().warn("Received goal message with insufficient data.")

    # Callback: Update the list of obstacles from the YOLO obstacles detection.
    def obstacles_callback(self, msg: Yolo):
        new_obstacles = []
        xywh_list = msg.xywh
        # Each obstacle is represented by 4 values: [x, y, width, height]. We take (x, y) as the obstacle’s location.
        for i in range(0, len(xywh_list), 4):
            obs_x = xywh_list[i]
            obs_y = xywh_list[i + 1]
            new_obstacles.append([obs_x, obs_y])
        self.obstacles = new_obstacles
        self.get_logger().info(f"Obstacles updated: {self.obstacles}")

    # Callback: Update the occupancy grid.
    def occupancy_grid_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        flat_data = msg.data
        # Convert the flat occupancy grid list into a 2D list (rows: height, columns: width)
        grid_2d = [list(flat_data[i * width:(i + 1) * width]) for i in range(height)]
        self.occupancy_grid = grid_2d
        self.get_logger().info("Occupancy grid updated.")

    # Callback: Update the planned path from the Path Planner.
    def path_json_callback(self, msg: BittlePathJSON):
        try:
            data = json.loads(msg.json_data)
            self.planned_path = data.get("path", [])
            self.get_logger().info(f"Planned path updated: {self.planned_path}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to decode planned path JSON: {e}")

    # Timer callback: aggregate data, send to GPT-4, and publish the command.
    def timer_callback(self):
        # Ensure that all necessary data is available.
        if (self.robot_position is None or 
            self.goal_position is None or 
            self.occupancy_grid is None or 
            self.planned_path is None):
            self.get_logger().info("Waiting for complete data for reasoning...")
            return

        # Build the aggregated JSON data structure.
        aggregated_data = {
            "robot_position": self.robot_position,
            "goal_position": self.goal_position,
            "obstacles": self.obstacles,
            "occupancy_grid": self.occupancy_grid,
            "planned_path": self.planned_path
        }
        input_json = json.dumps(aggregated_data)
        self.get_logger().info(f"Aggregated data: {input_json}")

        # Build a system prompt that explains what the LLM (GPT-4) should do.
        system_prompt = (
            "You are an intelligent robot control system. Based on the provided aggregated sensor data, "
            "generate a high-level command for the robot. The aggregated data includes the robot's current position, "
            "goal position, obstacle locations, an occupancy grid map, and a precomputed path. "
            "Provide a concise command in JSON format with a field 'command'. For example: "
            "{\"command\": \"move_forward\"}."
        )

        # Call the GPT-4 API (via OpenAI's ChatCompletion interface)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_json}
                ],
                max_tokens=50,
                temperature=0.2,
            )
            # Extract the GPT-4 reply
            gpt_reply = response['choices'][0]['message']['content']
            self.get_logger().info(f"GPT-4 response: {gpt_reply}")
        except Exception as e:
            self.get_logger().error(f"Error calling GPT-4 API: {e}")
            return

        # Parse GPT-4’s reply (which should be valid JSON with a 'command' field)
        try:
            command_data = json.loads(gpt_reply)
            command = command_data.get("command", "idle")
        except Exception as e:
            self.get_logger().error(f"Error parsing GPT-4 response: {e}")
            command = "idle"

        # Publish the command to the bittle_cmd topic.
        cmd_msg = String()
        # Wrap the command in JSON (this can be adapted later to a custom message type)
        cmd_msg.data = json.dumps({"command": command})
        self.cmd_pub.publish(cmd_msg)
        self.get_logger().info(f"Published command: {cmd_msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = LLMReasoning()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("LLM_Reasoning node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
