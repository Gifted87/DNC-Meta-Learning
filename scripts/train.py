import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import yaml
import argparse
import numpy as np
import os
import time
from tqdm import tqdm
from collections import deque

# Add project root to path to import models and environments
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary components
from environments import ProceduralMazeEnv, RepeatCopyEnv # Add other envs here
from models import DNC, LSTMBaseline

# Helper function for GAE calculation
@torch.no_grad() # Ensure no gradients are computed within this utility
def compute_gae_returns(rewards, values, dones, next_value, gamma, lambda_gae):
    """Computes Generalized Advantage Estimation (GAE) and returns.

    Args:
        rewards (list[float]): List of rewards for the trajectory segment.
        values (list[float]): List of value estimates V(s_t) for the segment.
        dones (list[bool]): List of done flags for the segment.
        next_value (float): Value estimate V(s_{t+N}) for the state after the last action.
        gamma (float): Discount factor.
        lambda_gae (float): GAE smoothing parameter.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Returns and Advantages tensors.
    """
    n_steps = len(rewards)
    advantages = torch.zeros(n_steps)
    last_gae_lam = 0.0
    values = values + [next_value] # Append V(s_{t+N})

    for t in reversed(range(n_steps)):
        mask = 1.0 - float(dones[t]) # Mask is 0 if done, 1 otherwise
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_gae_lam = delta + gamma * lambda_gae * mask * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + torch.tensor(values[:-1]) # R_t = A_t + V(s_t)
    return returns, advantages


def train(config_path: str, device_str: str = 'auto', run_id: Optional[str] = None):
    # --- Setup ---
    if device_str == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file {config_path}: {e}")
        return

    # Create unique run name and directories
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_type = config['model']['type']
    env_name = config['env']['type']
    if run_id is None:
        run_id = f"{env_name}_{model_type}_{timestamp}"
    else:
         run_id = f"{run_id}_{timestamp}" # Append timestamp even if ID provided

    log_dir = os.path.join(config['logging'].get('log_dir', 'runs'), run_id)
    save_dir = os.path.join(config['training'].get('save_dir', 'saved_models'), run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Log directory: {log_dir}")
    print(f"Save directory: {save_dir}")

    # Save config file for reproducibility
    try:
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"Warning: Could not save config file: {e}")


    # Environment Setup
    env_config = config['env']
    if env_name == 'ProceduralMazeEnv':
        env = ProceduralMazeEnv(
            width=env_config.get('width', 5),
            height=env_config.get('height', 5),
            max_steps=env_config.get('max_steps', 100)
        )
    elif env_name == 'RepeatCopyEnv':
        env = RepeatCopyEnv(
            max_seq_len=env_config.get('max_seq_len', 5),
            max_repeats=env_config.get('max_repeats', 3)
        )
    else:
        # Try gymnasium make for standard envs
        try:
            env = gym.make(env_name)
            # Try to get input/output sizes for gym envs
            if isinstance(env.observation_space, spaces.Box):
                 input_size = np.prod(env.observation_space.shape)
            # Add handling for other space types if needed
            else: raise NotImplementedError(f"Observation space {type(env.observation_space)} not handled yet for gym.make")

            if isinstance(env.action_space, spaces.Discrete):
                 num_actions = env.action_space.n
            else: raise NotImplementedError(f"Action space {type(env.action_space)} not handled yet for gym.make")

            print(f"Using Gym environment: {env_name}")

        except gym.error.NameNotFound:
             print(f"Error: Unknown environment type: {env_name}")
             return

    # Get input/output sizes (handle envs providing obs_size directly)
    if hasattr(env, 'obs_size'):
         input_size = env.obs_size
    elif isinstance(env.observation_space, spaces.Box):
         input_size = np.prod(env.observation_space.shape)
    else:
         raise NotImplementedError(f"Cannot determine input_size for obs space {type(env.observation_space)}")

    if isinstance(env.action_space, spaces.Discrete):
         num_actions = env.action_space.n
    else:
         raise NotImplementedError(f"Cannot determine num_actions for action space {type(env.action_space)}")


    # Model Setup
    model_config = config['model']
    hidden_size = model_config['hidden_size']

    if model_type == 'dnc':
        model = DNC(
            input_size=input_size,
            hidden_size=hidden_size,
            num_actions=num_actions,
            memory_slots=model_config.get('memory_slots', 64),
            memory_vector_size=model_config.get('memory_vector_size', 32),
            num_read_heads=model_config.get('num_read_heads', 3),
            k_sparse_read=model_config.get('k_sparse_read', None) # None defaults to ~5%
        ).to(device)
    elif model_type == 'lstm':
        model = LSTMBaseline(
            input_size=input_size,
            hidden_size=hidden_size,
            num_actions=num_actions
        ).to(device)
    else:
        print(f"Error: Unknown model type: {model_type}")
        env.close()
        return

    print(f"Model ({model_type}):")
    # print(model) # Can be very verbose for DNC
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")


    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        eps=config['training'].get('adam_eps', 1e-5) # Added Adam epsilon
    )

    # Training Parameters
    num_episodes = config['training']['num_episodes']
    gamma = config['training']['gamma']
    lambda_gae = config['training'].get('lambda_gae', 0.95)
    value_loss_coef = config['training'].get('value_loss_coef', 0.5)
    entropy_coef = config['training'].get('entropy_coef', 0.01)
    grad_clip_norm = config['training'].get('grad_clip_norm', 0.5)
    n_steps_update = config['training'].get('n_steps_update', 20) # Collect N steps before update
    save_interval = config['training'].get('save_interval', 100) # Save checkpoint every N episodes

    # Logging
    writer = SummaryWriter(log_dir)
    recent_rewards = deque(maxlen=100) # Track rewards of last 100 episodes

    print(f"Starting training for {num_episodes} episodes...")

    # --- Training Loop ---
    global_step = 0
    start_time = time.time()

    for episode in range(num_episodes):
        obs, info = env.reset()
        # Model state needs to be handled carefully for RNNs. Reset at episode start.
        # Assuming batch_size=1 for simplicity in this loop. Adapt if using >1 worker.
        state = model.init_state(batch_size=1, device=device)
        episode_reward = 0
        episode_length = 0
        done = False

        # Buffers for N-step A2C update
        ep_obs, ep_rewards, ep_actions, ep_dones, ep_log_probs, ep_values, ep_entropies = [], [], [], [], [], [], []
        # Store states for BPTT if needed, but simple A2C often updates based on buffers
        # ep_states = [] # Optional: Store states if BPTT over the n_steps_update sequence is desired


        while not done:
            # Convert observation to tensor, add batch dim, move to device
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)

            # Get action from model - Use torch.no_grad for action selection
            with torch.no_grad():
                # Need to detach state tuple elements if state is reused across updates
                # This is complex. Simpler: Re-run model from start of buffer if using BPTT.
                # For simple A2C, only need V(s_t) and pi(a_t|s_t) from the forward pass.
                detached_state = {k: (s[0].detach(), s[1].detach()) if isinstance(s, tuple) else s.detach()
                                  for k, s in state.items()}
                action_dist, value, next_state = model(obs_tensor, detached_state) # Use detached state

            action = action_dist.sample() # Sample action
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().item())
            done = terminated or truncated

            # Store experience (using detached tensors or numpy/python types)
            # Store obs *before* action was taken
            ep_obs.append(obs_tensor.cpu()) # Store tensor directly
            ep_rewards.append(reward)
            ep_actions.append(action.cpu()) # Store tensor
            ep_dones.append(done)
            ep_log_probs.append(log_prob.cpu()) # Store tensor
            ep_values.append(value.cpu().item()) # Store scalar value V(s_t)
            ep_entropies.append(entropy.cpu()) # Store tensor

            # Prepare for next step
            obs = next_obs
            state = next_state # Use the *non-detached* state for next step calculation
            episode_reward += reward
            episode_length += 1
            global_step += 1

            # --- A2C Update ---
            if len(ep_rewards) >= n_steps_update or done:
                # --- Calculate Returns and Advantages (GAE) ---
                with torch.no_grad():
                    # Get value of the *last* state S_{t+N} or final state if done
                    obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
                    detached_state = {k: (s[0].detach(), s[1].detach()) if isinstance(s, tuple) else s.detach()
                                      for k, s in state.items()} # Detach final state
                    _, bootstrap_value, _ = model(obs_tensor, detached_state)
                    final_value = bootstrap_value.cpu().item() * (1.0 - float(done)) # Zero if done

                returns, advantages = compute_gae_returns(
                    ep_rewards, ep_values, ep_dones, final_value, gamma, lambda_gae
                )
                returns = returns.to(device)
                advantages = advantages.to(device)

                # Normalize advantages (optional but recommended)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # --- Prepare batch for loss calculation ---
                # No BPTT needed if we just use the stored log_probs and values
                log_probs_tensor = torch.stack(ep_log_probs).to(device)
                values_tensor = torch.tensor(ep_values, dtype=torch.float32).to(device) # Already calculated V(s_t)
                entropies_tensor = torch.stack(ep_entropies).to(device)

                # --- Calculate Losses ---
                # Policy Loss (negative sign because we want to maximize log_prob * advantage)
                policy_loss = -(log_probs_tensor * advantages).mean()

                # Value Loss (Smooth L1 or MSE between V(s_t) and calculated Returns R_t)
                value_loss = F.smooth_l1_loss(values_tensor, returns).mean() # Or F.mse_loss
                # value_loss = F.mse_loss(values_tensor, returns).mean()

                # Entropy Loss (negative sign because we want to maximize entropy)
                entropy_loss = -entropies_tensor.mean()

                # Total Loss
                total_loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss

                # --- Optimization Step ---
                optimizer.zero_grad()
                total_loss.backward() # Gradients flow back through stored log_probs/values
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

                # --- Log losses ---
                writer.add_scalar('Loss/Total', total_loss.item(), global_step)
                writer.add_scalar('Loss/Policy', policy_loss.item(), global_step)
                writer.add_scalar('Loss/Value', value_loss.item(), global_step)
                writer.add_scalar('Loss/Entropy', entropy_loss.item(), global_step)

                # Clear buffers for next N steps
                ep_obs, ep_rewards, ep_actions, ep_dones, ep_log_probs, ep_values, ep_entropies = [], [], [], [], [], [], []


        # --- End of Episode Logging ---
        recent_rewards.append(episode_reward)
        avg_reward_100 = np.mean(recent_rewards) if len(recent_rewards) == 100 else np.mean(list(recent_rewards))

        if (episode + 1) % 10 == 0: # Print every 10 episodes
            elapsed_time = time.time() - start_time
            print(f"Episode {episode+1}/{num_episodes} | Step: {global_step} | Length: {episode_length} | "
                  f"Reward: {episode_reward:.2f} | Avg Rwd (100): {avg_reward_100:.2f} | Time: {elapsed_time:.1f}s")

        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Length', episode_length, episode)
        writer.add_scalar('Episode/AvgReward100', avg_reward_100, episode)

        # --- Save Checkpoint ---
        if (episode + 1) % save_interval == 0 or (episode + 1) == num_episodes:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_ep{episode+1}.pt")
            torch.save({
                'episode': episode + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # Add any other state needed for resuming (e.g., RNG state)
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


    # --- Cleanup ---
    env.close()
    writer.close()
    print("Training finished.")
    print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DNC or LSTM model using A2C.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use (default: auto).")
    parser.add_argument("--run_id", type=str, default=None, help="Custom ID for the training run (optional).")
    args = parser.parse_args()

    train(config_path=args.config, device_str=args.device, run_id=args.run_id)
