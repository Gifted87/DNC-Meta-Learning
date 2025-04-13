# Benchmark Results

This document will contain the results of benchmarking the DNC model against the LSTM baseline on various tasks.

**Goal:** Quantify the performance difference (e.g., sample efficiency, final performance, generalization) between DNC and LSTM, particularly on tasks requiring structured memory or long-range dependencies. The initial claim was "40% fewer episodes than LSTM baselines" on specific procedural tasks.

## Experimental Setup

- **Environments:**
    - `ProceduralMazeEnv` (configs/dnc.yaml, configs/lstm.yaml)
    - `RepeatCopyEnv` (TODO: Add specific configs)
    - *(Add other environments as tested)*
- **Models:** DNC, LSTM
- **Training Algorithm:** A2C (as implemented in `scripts/train.py`)
- **Key Hyperparameters:** See corresponding YAML config files saved in the `runs/<run_id>/` directory.
- **Metrics:**
    - Average episode reward over training (smoothed).
    - Average episode length over training (smoothed).
    - Success rate or task-specific metric (e.g., % correct bits in RepeatCopy).
    - Number of episodes/steps to reach a target performance threshold.

## Results Summary Table (Placeholder)

| Environment         | Model | Avg. Episodes to Target | Final Performance (Metric) | Notes                       | Run ID / Link to Logs      |
|---------------------|-------|-------------------------|----------------------------|-----------------------------|----------------------------|
| ProceduralMaze-5x5  | LSTM  | *TBD*                   | *TBD* (Avg Reward)         | Baseline                    | *Link to runs/*            |
| ProceduralMaze-5x5  | DNC   | ***TBD***               | *TBD* (Avg Reward)         | Target: ~40% fewer episodes | *Link to runs/*            |
| RepeatCopy-L5-R3    | LSTM  | *TBD*                   | *TBD* (% Correct)          | Baseline                    | *Link to runs/*            |
| RepeatCopy-L5-R3    | DNC   | *TBD*                   | *TBD* (% Correct)          | Expect significant advantage| *Link to runs/*            |
| *(Add more rows)*   | ...   | ...                     | ...                        | ...                         | ...                        |

