
#!/usr/bin/env python3
# deploy/ros_integration_example.py

# Conceptual Example: Integrating the DNC/LSTM model into a ROS 2 node.
# This assumes you have a trained model checkpoint and the necessary ROS 2 setup.
# It does NOT include the actual ROS 2 node setup, message definitions, etc.

import rclpy
from rclpy.node import Node
# Replace with your actual message types
from std_msgs.msg import Float32MultiArray # Example for observation
from std_msgs.msg import Int32 # Example for action

import torch
import numpy as np
import yaml
import os

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DNC, LSTMBaseline # Import your model classes

class RLAgentNode(Node):

    def __init__(self, config_path, checkpoint_path, device_str='cpu'):
        super().__init__('rl_agent_node')
        self.get_logger().info('Initializing RL Agent Node...')

        self.device = torch.device(device_str)

        # --- Load Config and Model (Similar to visualize.py) ---
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load config {config_path}: {e}")
            raise e # Or handle gracefully

        self.model_type = self.config['model']['type']
        # Determine input/output sizes (needs robust way, maybe from config/checkpoint)
        # Placeholder values - replace with actual logic
        input_size = self.config.get('input_size_placeholder', 64) # Add to config or derive
        num_actions = self.config.get('num_actions_placeholder', 4) # Add to config or derive

        if self.model_type == 'dnc':
            self.model = DNC(input_size, self.config['model']['hidden_size'], num_actions, **self.config['model']).to(self.device) # Pass relevant model args
        elif self.model_type == 'lstm':
            self.model = LSTMBaseline(input_size, self.config['model']['hidden_size'], num_actions).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.get_logger().info(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise e

        # --- Initialize Model State ---
        self.model_state = self.model.init_state(batch_size=1, device=self.device) # Batch size 1 for inference
        self.get_logger().info('Model state initialized.')

        # --- ROS Subscribers and Publishers ---
        # Replace 'observation_topic' and 'action_topic' with actual topic names
        # Replace Float32MultiArray and Int32 with actual message types
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'observation_topic',
            self.observation_callback,
            10) # QoS profile depth
        self.publisher_ = self.create_publisher(Int32, 'action_topic', 10)

        self.get_logger().info('ROS Subscribers and Publishers created.')
        self.get_logger().info(f'{self.get_name()} node started successfully.')


    def observation_callback(self, msg):
        """Callback function for receiving observations."""
        self.get_logger().debug('Received observation')

        # --- Preprocess Observation ---
        # Convert ROS message to numpy array, then to torch tensor
        # The exact conversion depends on your observation message format
        try:
            # Example: assuming Float32MultiArray.data is a flat list/array
            obs_np = np.array(msg.data, dtype=np.float32)
            # Validate shape if possible (needs input_size)
            # obs_np = obs_np.reshape(...) # Reshape if needed
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device) # Add batch dim
        except Exception as e:
            self.get_logger().error(f"Error processing observation data: {e}")
            return

        # --- Model Inference ---
        with torch.no_grad():
            try:
                # Get action distribution, value (optional), and next state
                action_dist, _, self.model_state = self.model(obs_tensor, self.model_state)

                # Choose action (e.g., sample or take argmax for deterministic policy)
                action = action_dist.sample()
                # action = torch.argmax(action_dist.probs, dim=-1) # Deterministic action
                action_item = action.cpu().item()

            except Exception as e:
                self.get_logger().error(f"Error during model inference: {e}")
                # Optionally reset state on error?
                # self.model_state = self.model.init_state(batch_size=1, device=self.device)
                return

        # --- Publish Action ---
        action_msg = Int32()
        action_msg.data = int(action_item) # Ensure correct type for message
        self.publisher_.publish(action_msg)
        self.get_logger().debug(f'Published action: {action_item}')

        # Potential: Add logic to reset model state based on an 'episode_done' topic


def main(args=None):
    rclpy.init(args=args)

    # --- Node Configuration ---
    # TODO: Replace with your actual paths, load from params or args
    config_path = '/path/to/your/config.yaml'
    checkpoint_path = '/path/to/your/checkpoint.pt'
    device = 'cpu' # Or 'cuda'

    # Basic check if paths exist (replace with proper ROS parameter handling)
    if not os.path.exists(config_path):
         print(f"Error: Config path not found: {config_path}")
         return
    if not os.path.exists(checkpoint_path):
         print(f"Error: Checkpoint path not found: {checkpoint_path}")
         return

    try:
        agent_node = RLAgentNode(config_path, checkpoint_path, device)
        rclpy.spin(agent_node)
    except KeyboardInterrupt:
        print("Node shutting down...")
    except Exception as e:
        print(f"Node failed to initialize or run: {e}")
    finally:
        # Cleanup
        if 'agent_node' in locals() and agent_node:
            agent_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    # Note: This script needs to be integrated into a ROS 2 package structure
    #       and launched appropriately (e.g., using ros2 run or a launch file).
    print("--- Conceptual ROS 2 DNC/LSTM Agent Node ---")
    print("Ensure config_path and checkpoint_path are correctly set in the main function.")
    print("Run this within a sourced ROS 2 environment.")
    main()
