# Differentiable Neural Computer (DNC) for Meta-Learning & RL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Core Idea:** An RL agent augmented with external, addressable, differentiable memory, designed to solve complex sequential tasks and potentially exhibit faster adaptation (meta-learning properties). This implementation aims to replicate core concepts from DeepMind's DNC paper within an RL framework.

**Target Claim:** Solve certain procedural tasks (e.g., mazes) with significantly fewer episodes (~40% goal) compared to standard LSTM baselines by leveraging learned memory access patterns.

## Key Features & Innovations

🧠 **External Differentiable Memory:**
   - Controller (LSTM) interacts with an `N x M` memory matrix.
   - Attention-based **Read Heads** using sparse content addressing (Top-K).
   - Attention-based **Write Head** using content *and* allocation-based addressing (writing to unused slots).
   - Implements **Usage Tracking** and **Temporal Link Matrix** updates as per the DNC paper.

📉 **Reinforcement Learning Integration:**
   - Trained using **Actor-Critic (A2C)** with Generalized Advantage Estimation (GAE).
   - Compatible with Gymnasium environments.
   - Includes `ProceduralMazeEnv` and `RepeatCopyEnv` examples.

📊 **Benchmarking Framework:**
   - Includes an LSTM baseline for comparison.
   - Config-driven training (`scripts/train.py`) via YAML files.
   - Logging via TensorBoard (`runs/` directory).
   - Tools for memory visualization (`scripts/visualize.py`).

## Project Structure

```plaintext
dnc-meta-learning/
├── environments/      # Gymnasium environment implementations
│   ├── __init__.py
│   ├── maze_env.py    # Procedural maze navigation
│   └── algo_env.py    # Algorithmic tasks (e.g., RepeatCopy)
├── models/            # PyTorch model definitions
│   ├── __init__.py
│   ├── dnc.py         # Core DNC architecture
│   ├── lstm_baseline.py # Baseline comparison model
│   ├── memory_heads.py# Attention-based read/write heads
│   └── utils.py       # Helper functions (cosine similarity, etc.)
├── tests/             # Unit tests
│   ├── __init__.py
│   └── test_memory.py # Tests for memory utility functions
├── scripts/           # Executable scripts
│   ├── __init__.py
│   ├── train.py       # Main training loop (A2C)
│   └── visualize.py   # Generate memory access visualizations
├── configs/           # Hyperparameter configuration files
│   ├── dnc.yaml
│   └── lstm.yaml
├── docs/              # Documentation and results
│   ├── MEMORY.md      # Details on the DNC memory mathematics
│   └── RESULTS.md     # Placeholder for benchmark results and graphs
├── deploy/            # Deployment examples (Experimental)
│   ├── export_onnx.py # Script to export model to ONNX (TODO)
│   └── ros_integration_example.py # Conceptual ROS integration (TODO)
├── saved_models/      # Default location for saved model checkpoints
├── runs/              # Default location for TensorBoard logs
└── README.md          # This file```

## Quick Start

**1. Clone & Setup Environment:**

```bash
git clone https://github.com/your-username/dnc-meta-learning.git # TODO: Update URL
cd dnc-meta-learning

# Create conda environment (recommended)
conda create -n dnc python=3.9
conda activate dnc

# Install dependencies
# Adjust PyTorch command based on your CUDA version or use CPU-only
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Example for CUDA 11.8
pip install gymnasium numpy pyyaml tqdm tensorboard matplotlib # Add PyYAML, tqdm, tensorboard, matplotlib

# For visualization saving (GIF):
# sudo apt-get update && sudo apt-get install imagemagick # (Linux)
# OR ensure Pillow is sufficient (pip install Pillow)
```

**2. Train Models:**

```bash
# Train DNC on the default maze
python scripts/train.py --config configs/dnc.yaml --device auto

# Train LSTM baseline on the default maze
python scripts/train.py --config configs/lstm.yaml --device auto

# Train on RepeatCopy task (adjust config files first)
# python scripts/train.py --config configs/dnc_copy.yaml --device auto
```
*Training logs and checkpoints will be saved in `runs/` and `saved_models/` respectively, organized by run ID.*

**3. Monitor Training (Optional):**

```bash
tensorboard --logdir runs/
```
*Open your browser to the provided URL (usually `http://localhost:6006`).*

**4. Visualize Memory Access (After Training):**

```bash
# Find a checkpoint file, e.g., saved_models/ProceduralMazeEnv_dnc_.../checkpoint_ep5000.pt
python scripts/visualize.py \
    --config runs/ProceduralMazeEnv_dnc_.../config.yaml `# Path to the config used for the run` \
    --checkpoint saved_models/ProceduralMazeEnv_dnc_.../checkpoint_ep5000.pt \
    --output maze_dnc_memory.gif \
    --device cpu
```

## Potential Applications

*   **Robotics:** Learning complex manipulation sequences, long-horizon planning with persistent state.
*   **Algorithm Learning:** Solving simple programs, graph traversal, or sequence manipulation tasks.
*   **Natural Language Processing:** Story understanding, question answering requiring reasoning over long contexts.
*   **Meta-Learning:** The memory could potentially store task-specific information, allowing faster adaptation to new variations of a task.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) filefor details.

## References

*   Graves, A., Wayne, G., Reynolds, M. et al. Hybrid computing using a neural network with dynamic external memory. Nature 538, 471–476 (2016). [https://doi.org/10.1038/nature20101](https://doi.org/10.1038/nature20101)
