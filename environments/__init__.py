# environments/__init__.py

# Make environments accessible
from .maze_env import ProceduralMazeEnv
from .algo_env import RepeatCopyEnv

# Optional: Register environments with Gymnasium if needed globally
# import gymnasium as gym
# gym.register(id='ProceduralMaze-v0', entry_point='environments.maze_env:ProceduralMazeEnv')
# gym.register(id='RepeatCopy-v0', entry_point='environments.algo_env:RepeatCopyEnv')

__all__ = ['ProceduralMazeEnv', 'RepeatCopyEnv']