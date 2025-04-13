import torch
import torch.onnx
import yaml
import argparse
import os
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DNC, LSTMBaseline

# Warning: Exporting RNNs with state and complex logic like DNC memory
#          to ONNX can be very challenging and may not capture the full dynamics
#          or require specific model adaptations (e.g., using LSTM layer instead of cell).
#          This script provides a basic structure and is likely to fail or produce
#          a limited ONNX graph without significant effort.

def export_to_onnx(config_path: str, checkpoint_path: str, output_onnx_path: str, device_str: str = 'cpu'):
    """Attempts to export a trained model to ONNX format."""

    device = torch.device(device_str)
    print(f"Using device: {device}")

    # --- Load Config ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    model_type = config['model']['type']
    env_config = config['env']
    env_name = env_config['type']

    # --- Determine Dummy Input Size (Crucial for ONNX) ---
    # Need a fixed input size for tracing. This is problematic for variable envs.
    # Using sizes from config assuming they are representative.
    # TODO: Make this more robust based on actual env space if possible
    if env_name == 'ProceduralMazeEnv':
        grid_height = env_config.get('height', 5) * 2 + 1
        grid_width = env_config.get('width', 5) * 2 + 1
        input_size = grid_height * grid_width
    elif env_name == 'RepeatCopyEnv':
        input_size = 4 # From RepeatCopyEnv definition
    else:
        # Need logic to get input size for other envs (e.g., from gym.make)
        print(f"Warning: Cannot reliably determine input size for env type {env_name} from config alone.")
        # Fallback or raise error
        input_size = model_config.get('fallback_input_size', 64) # Add fallback to config?

    num_actions = model_config.get('num_actions', 4) # Need num_actions if loading model fails

    # --- Load Model ---
    model_config = config['model']
    hidden_size = model_config['hidden_size']

    if model_type == 'dnc':
        model = DNC(
            input_size=input_size,
            hidden_size=hidden_size,
            num_actions=num_actions, # Provide num_actions here
            memory_slots=model_config.get('memory_slots', 64),
            memory_vector_size=model_config.get('memory_vector_size', 32),
            num_read_heads=model_config.get('num_read_heads', 3),
            k_sparse_read=model_config.get('k_sparse_read', None)
        ).to(device)
        # DNC state is complex - need dummy state for export
        dummy_state = model.init_state(batch_size=1, device=device)

    elif model_type == 'lstm':
        model = LSTMBaseline(
            input_size=input_size,
            hidden_size=hidden_size,
            num_actions=num_actions # Provide num_actions here
        ).to(device)
        dummy_state = model.init_state(batch_size=1, device=device)
    else:
        print(f"Error: Unknown model type: {model_type}")
        return

    # Load weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # --- Prepare Dummy Inputs ---
    # Batch size should typically be 1 for export unless dynamic axes are used well
    dummy_input = torch.randn(1, input_size, device=device)

    # State handling for ONNX is very tricky. LSTMCell state (tuple) is often problematic.
    # DNC state (dict of tensors/tuples) is even harder.
    # We might need to flatten the state or only export the stateless part.
    # **Attempt 1: Try exporting with state tuple (likely fails for DNC)**
    if model_type == 'lstm':
         dummy_state_input = dummy_state['controller'] # Tuple (h, c)
         input_names = ['input', 'h_in', 'c_in']
         output_names = ['action_probs', 'value', 'h_out', 'c_out'] # Assuming model outputs state
         # Need to modify LSTM model forward to return state explicitly for ONNX
         print("Warning: LSTM baseline forward needs modification to return state tuple for ONNX export.")

    elif model_type == 'dnc':
         # Flattening DNC state or handling dicts in ONNX is non-trivial.
         # This export will likely fail or be incomplete.
         print("Error: Exporting DNC state to ONNX is highly complex and not fully supported by this script.")
         print("Consider exporting only the controller or using a different approach.")
         # Example placeholder if trying tuple export (will fail):
         # dummy_state_input = tuple(v for v in dummy_state.values())
         # input_names = ['input'] + list(dummy_state.keys()) # Very complex input names
         # output_names = ['action_probs', 'value'] + list(dummy_state.keys()) # Outputting full state
         return # Abort DNC export for now


    # --- Export to ONNX ---
    print(f"Attempting to export model to {output_onnx_path}...")
    try:
        # torch.onnx.export might require specific opset_version
        # dynamic_axes might be needed for variable batch or sequence length (not handled here)
        torch.onnx.export(
            model,
            (dummy_input, dummy_state_input), # Model inputs must match forward signature
            output_onnx_path,
            export_params=True,
            opset_version=11, # Or newer, check compatibility
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            # dynamic_axes={'input': {0: 'batch_size'}, ... } # Example if needed
        )
        print("ONNX export completed (basic structure).")
        print("Validation with onnxruntime is recommended.")

    except Exception as e:
        print(f"\nError during ONNX export: {e}")
        print("Exporting models with complex state (LSTMCells, DNC memory) is challenging.")
        print("Consider model simplification or using TorchScript instead.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX format (Experimental).")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file used for training.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--output", type=str, default="model_export.onnx", help="Path to save the output ONNX file.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to load the model on.")
    args = parser.parse_args()

    if not args.output.endswith(".onnx"):
         args.output += ".onnx"

    export_to_onnx(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        device_str=args.device
    )

