import torch
import gymnasium as gym
import yaml
import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments import ProceduralMazeEnv, RepeatCopyEnv
from models import DNC # Assuming visualization focuses on DNC memory

def visualize_memory(config_path: str, checkpoint_path: str, output_gif: str, device_str: str = 'cpu'):
    """Loads a trained DNC model and visualizes memory access patterns during an episode."""

    device = torch.device(device_str)
    print(f"Using device: {device}")

    # --- Load Config ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing config file {config_path}: {e}")
        return

    if config['model']['type'] != 'dnc':
        print("Error: Visualization script currently only supports DNC models.")
        return

    # --- Setup Environment (mirrors training setup) ---
    env_config = config['env']
    env_name = env_config['type']
    if env_name == 'ProceduralMazeEnv':
        env = ProceduralMazeEnv(
            width=env_config.get('width', 5),
            height=env_config.get('height', 5),
            max_steps=env_config.get('max_steps', 100),
            render_mode='rgb_array' # Enable RGB rendering for visualization
        )
    elif env_name == 'RepeatCopyEnv':
         # TODO: Add simple visualization for AlgoEnv if needed
        print("Warning: Visualization for RepeatCopyEnv not implemented yet.")
        env = RepeatCopyEnv(
            max_seq_len=env_config.get('max_seq_len', 5),
            max_repeats=env_config.get('max_repeats', 3)
        )
        # Algo env doesn't have standard rendering, skip frame saving later
    else:
        # Add other env loading logic if needed
        print(f"Error: Unknown or unsupported environment type for visualization: {env_name}")
        return

    # Get input/output sizes
    if hasattr(env, 'obs_size'):
         input_size = env.obs_size
    elif isinstance(env.observation_space, spaces.Box):
         input_size = np.prod(env.observation_space.shape)
    else: raise NotImplementedError("Cannot determine input_size")
    if isinstance(env.action_space, spaces.Discrete):
         num_actions = env.action_space.n
    else: raise NotImplementedError("Cannot determine num_actions")


    # --- Load Model ---
    model_config = config['model']
    model = DNC(
        input_size=input_size,
        hidden_size=model_config['hidden_size'],
        num_actions=num_actions,
        memory_slots=model_config.get('memory_slots', 64),
        memory_vector_size=model_config.get('memory_vector_size', 32),
        num_read_heads=model_config.get('num_read_heads', 3),
        k_sparse_read=model_config.get('k_sparse_read', None)
    ).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set model to evaluation mode
        print(f"Model loaded from {checkpoint_path} (trained for {checkpoint.get('episode', 'N/A')} episodes)")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        env.close()
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        env.close()
        return

    # --- Run Episode and Collect Data ---
    obs, info = env.reset()
    state = model.init_state(batch_size=1, device=device)
    done = False
    episode_reward = 0
    step_count = 0

    # Data storage for visualization
    memory_states = [] # List to store memory matrices (optional)
    read_weights_history = [] # List of read weights (B, num_heads, N) per step
    write_weights_history = [] # List of write weights (B, N) per step
    usage_history = [] # List of usage vectors (B, N) per step
    env_frames = [] # List of env renderings

    print("Running episode to collect memory access data...")
    with torch.no_grad():
        while not done:
            # Get env frame BEFORE action
            if hasattr(env, 'render') and env.render_mode == 'rgb_array':
                 frame = env.render()
                 if frame is not None:
                     env_frames.append(frame)

            # Prepare observation
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)

            # Model forward pass
            action_dist, value, state = model(obs_tensor, state)
            action = action_dist.sample() # Or action = action_dist.probs.argmax()

            # --- Store memory-related state for visualization ---
            # Detach tensors before storing to avoid holding onto graph
            # memory_states.append(state['memory'].squeeze(0).cpu().numpy()) # Can be large!
            read_weights_history.append(state['read_weights'].squeeze(0).cpu().numpy())
            write_weights_history.append(state['write_weights'].squeeze(0).cpu().numpy())
            usage_history.append(state['usage'].squeeze(0).cpu().numpy())

            # Environment step
            obs, reward, terminated, truncated, info = env.step(action.cpu().item())
            done = terminated or truncated
            episode_reward += reward
            step_count += 1

            if step_count > env_config.get('max_steps', 100) * 1.5: # Safety break
                print("Warning: Episode seems too long, breaking.")
                break

    env.close()
    print(f"Episode finished. Length: {step_count}, Reward: {episode_reward:.2f}")
    print(f"Collected {len(read_weights_history)} steps of memory data.")

    if not read_weights_history:
        print("Error: No memory data collected.")
        return

    # --- Create Animation ---
    num_steps = len(read_weights_history)
    num_read_heads = read_weights_history[0].shape[0]
    memory_slots = read_weights_history[0].shape[1]

    fig, axes = plt.subplots(2 + num_read_heads, 1, figsize=(6, 3 * (2 + num_read_heads)), sharex=True)
    fig.suptitle(f'DNC Memory Access - Run: {os.path.basename(checkpoint_path)}')

    # Add subplot for environment frame if available
    ax_env = None
    if env_frames and len(env_frames) == num_steps:
         # Adjust layout if env frame is present
         fig.clf() # Clear current figure setup
         grid_spec_rows = 2 + num_read_heads
         fig = plt.figure(figsize=(8, 2 + 1.5 * grid_spec_rows)) # Wider figure
         gs = fig.add_gridspec(grid_spec_rows, 2, width_ratios=[3, 1]) # Env on left, memory on right

         ax_env = fig.add_subplot(gs[:, 0]) # Env frame spans all rows on the left
         mem_axes = [fig.add_subplot(gs[i, 1]) for i in range(grid_spec_rows)]
         axes = mem_axes # Use these axes for memory plots
         ax_env.set_title("Environment State")
         ax_env.axis('off')
         env_im = ax_env.imshow(env_frames[0])
    else:
         env_frames = None # Ensure it's None if not usable


    # Prepare memory axes
    im_reads = []
    for i in range(num_read_heads):
        ax = axes[i]
        im = ax.imshow(read_weights_history[0][i:i+1, :], cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Read Head {i+1} Weights')
        ax.set_yticks([])
        im_reads.append(im)

    im_write = axes[num_read_heads].imshow(write_weights_history[0][np.newaxis, :], cmap='plasma', aspect='auto', vmin=0, vmax=1)
    axes[num_read_heads].set_title('Write Head Weights')
    axes[num_read_heads].set_yticks([])

    im_usage = axes[num_read_heads + 1].imshow(usage_history[0][np.newaxis, :], cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    axes[num_read_heads + 1].set_title('Memory Usage (1=Used, 0=Free)')
    axes[num_read_heads + 1].set_yticks([])
    axes[num_read_heads + 1].set_xlabel('Memory Slot Index')


    # Add colorbars (optional, can clutter)
    # fig.colorbar(im_reads[0], ax=axes[0], orientation='horizontal', fraction=0.1, pad=0.2)
    # fig.colorbar(im_write, ax=axes[num_read_heads], orientation='horizontal', fraction=0.1, pad=0.2)
    # fig.colorbar(im_usage, ax=axes[num_read_heads + 1], orientation='horizontal', fraction=0.1, pad=0.2)

    time_text = fig.text(0.5, 0.01, '', ha='center') # Add text for step number

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    def update(frame):
        if env_frames:
             env_im.set_data(env_frames[frame])

        for i in range(num_read_heads):
            im_reads[i].set_data(read_weights_history[frame][i:i+1, :])
        im_write.set_data(write_weights_history[frame][np.newaxis, :])
        im_usage.set_data(usage_history[frame][np.newaxis, :])
        time_text.set_text(f'Step: {frame + 1}/{num_steps}')

        artists = im_reads + [im_write, im_usage, time_text]
        if env_frames: artists.append(env_im)
        return artists


    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=150, blit=True) # Adjust interval for speed

    # Save animation
    try:
        print(f"Saving animation to {output_gif}...")
        ani.save(output_gif, writer='imagemagick', fps=10) # Or use 'pillow' if imagemagick not installed
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Ensure ffmpeg or imagemagick is installed and available in your PATH.")
        print("Alternatively, try saving with writer='pillow'.")
        # Try pillow as fallback
        try:
             print("Trying to save with Pillow writer...")
             ani.save(output_gif, writer='pillow', fps=10)
             print("Animation saved successfully with Pillow.")
        except Exception as e2:
             print(f"Error saving animation with Pillow: {e2}")


    plt.close(fig) # Close the plot window


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DNC memory access patterns.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file used for training.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--output", type=str, default="dnc_memory_visualization.gif", help="Path to save the output GIF.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run the model on.")
    args = parser.parse_args()

    visualize_memory(config_path=args.config, checkpoint_path=args.checkpoint, output_gif=args.output, device_str=args.device)
