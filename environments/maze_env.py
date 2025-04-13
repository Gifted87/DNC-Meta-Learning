# environments/maze_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import heapq  # For A* path check (optional but good)
# Optional rendering dependency
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. RGB rendering will not be available for ProceduralMazeEnv.")

class MazeGenerator:
    """Generates mazes using Randomized Prim's algorithm."""
    def __init__(self, width, height):
        if width < 2 or height < 2:
             raise ValueError("Maze dimensions must be at least 2x2")
        self.width = width
        self.height = height
        self.maze = np.ones((height * 2 + 1, width * 2 + 1), dtype=np.uint8) # 1 = wall

    def _is_valid(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width

    def generate(self):
        start_node = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
        r, c = start_node
        self.maze[r * 2 + 1, c * 2 + 1] = 0 # Mark starting cell as path

        frontiers = []
        # Add neighbors of the start cell to the frontier list
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            wall_r, wall_c = r * 2 + 1 + dr, c * 2 + 1 + dc
            cell_r, cell_c = nr * 2 + 1, nc * 2 + 1
            if self._is_valid(nr, nc):
                frontiers.append(((nr, nc), (wall_r, wall_c), (cell_r, cell_c))) # (NeighborCell, WallToBreak, NeighborGridPos)

        while frontiers:
            idx = random.randrange(len(frontiers))
            (nr, nc), (wall_r, wall_c), (cell_r, cell_c) = frontiers.pop(idx)

            # If the neighbor cell is already part of the maze (a path), skip
            if self.maze[cell_r, cell_c] == 0:
                continue

            # Connect the frontier cell to the maze
            self.maze[wall_r, wall_c] = 0
            self.maze[cell_r, cell_c] = 0

            # Add new frontiers from the newly added cell
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nnr, nnc = nr + dr, nc + dc
                n_wall_r, n_wall_c = cell_r + dr, cell_c + dc
                n_cell_r, n_cell_c = nnr * 2 + 1, nnc * 2 + 1
                if self._is_valid(nnr, nnc) and self.maze[n_cell_r, n_cell_c] == 1:
                     frontiers.append(((nnr, nnc), (n_wall_r, n_wall_c), (n_cell_r, n_cell_c)))

        # Ensure start and goal are open passages
        self.maze[1, 1] = 0
        self.maze[self.height * 2 -1, self.width * 2 - 1] = 0

        # Sometimes Prim's can isolate the goal, ensure connectivity (simple check)
        if self.maze[self.height * 2 - 1, self.width * 2 - 2] == 1 and \
           self.maze[self.height * 2 - 2, self.width * 2 - 1] == 1:
             # If goal cell is walled off, open one wall randomly
             if random.random() < 0.5:
                  self.maze[self.height * 2 - 1, self.width * 2 - 2] = 0
             else:
                  self.maze[self.height * 2 - 2, self.width * 2 - 1] = 0

        return self.maze

class ProceduralMazeEnv(gym.Env):
    """
    A procedural maze environment using Gymnasium API.
    Observation is a flattened grid where: 0=path, 1=wall, 2=agent, 3=goal.
    Action space is Discrete(4): 0=Up, 1=Down, 2=Left, 3=Right.
    """
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 10}

    def __init__(self, width=5, height=5, max_steps=100, render_mode=None):
        super().__init__()
        if width < 2 or height < 2:
             raise ValueError("Maze dimensions must be at least 2x2")
        self.width = width
        self.height = height
        self._max_steps = max_steps
        self.action_space = spaces.Discrete(4) # 0: Up, 1: Down, 2: Left, 3: Right

        self.grid_height = height * 2 + 1
        self.grid_width = width * 2 + 1
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(self.grid_height * self.grid_width,), dtype=np.uint8
        )
        self.obs_size = self.grid_height * self.grid_width # Provide obs_size attribute

        self._action_to_delta = { # Grid coordinates (row, col)
            0: (-1, 0), # Up
            1: (1, 0),  # Down
            2: (0, -1), # Left
            3: (0, 1),  # Right
        }
        self._maze_generator = MazeGenerator(width, height)
        self._maze = None
        self._agent_location = None
        self._target_location = None
        self._step_count = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None # For pygame or matplotlib visualization
        self.clock = None # For pygame

        # Optional: A* check for solvability (can be slow for large mazes)
        self.check_solvability = False

    def _is_solvable(self, start, goal):
        """Check if a path exists using A*."""
        if self._maze is None: return False
        q = [(0 + self._heuristic(start, goal), 0, start)] # (f_cost, g_cost, pos)
        visited = {start}
        while q:
            f, g, pos = heapq.heappop(q)
            if pos == goal:
                return True
            for action in range(4):
                dr, dc = self._action_to_delta[action]
                next_pos = (pos[0] + dr, pos[1] + dc)
                if 0 <= next_pos[0] < self.grid_height and 0 <= next_pos[1] < self.grid_width and \
                   self._maze[next_pos] == 0 and next_pos not in visited:
                    visited.add(next_pos)
                    new_g = g + 1
                    h = self._heuristic(next_pos, goal)
                    heapq.heappush(q, (new_g + h, new_g, next_pos))
        return False

    def _heuristic(self, pos1, pos2):
        """Manhattan distance heuristic for A*."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_obs(self):
        obs_grid = self._maze.copy()
        obs_grid[self._agent_location] = 2 # Mark agent
        obs_grid[self._target_location] = 3 # Mark target
        return obs_grid.flatten()

    def _get_info(self):
        return {"distance": self._heuristic(self._agent_location, self._target_location)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Use the internal RNG if seeded
        rng = self.np_random if seed is not None else random

        while True: # Ensure generated maze is solvable if check is enabled
            self._maze = self._maze_generator.generate()
            self._agent_location = (1, 1) # Start top-left passage
            self._target_location = (self.grid_height - 2, self.grid_width - 2) # Goal bottom-right passage
            if not self.check_solvability or self._is_solvable(self._agent_location, self._target_location):
                break
            # print("Generated unsolvable maze, retrying...") # Debug

        self._step_count = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "rgb_array":
             self._render_frame()

        return observation, info

    def step(self, action):
        if self._maze is None:
             raise RuntimeError("Must call reset before step")

        delta_r, delta_c = self._action_to_delta[action]
        potential_r, potential_c = self._agent_location[0] + delta_r, self._agent_location[1] + delta_c

        terminated = False
        reward = -0.01 # Small step penalty to encourage efficiency

        # Check if move is valid (within bounds and not into a wall)
        if 0 <= potential_r < self.grid_height and 0 <= potential_c < self.grid_width and self._maze[potential_r, potential_c] == 0:
             self._agent_location = (potential_r, potential_c)
        # else: reward -= 0.05 # Optional: Larger penalty for hitting wall

        # Check if goal reached
        if self._agent_location == self._target_location:
            terminated = True
            reward += 1.0 # Goal reward

        self._step_count += 1
        truncated = self._step_count >= self._max_steps

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "rgb_array":
            self._render_frame()

        # Ensure reward is float32 for consistency if needed later
        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb() # Returns the frame buffer

    def _render_ansi(self):
        if self._maze is None: return ""
        grid = self._maze.copy()
        grid[self._agent_location] = 2
        grid[self._target_location] = 3
        render_str = ""
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if grid[r, c] == 1: render_str += "#" # Wall
                elif grid[r, c] == 0: render_str += " " # Path
                elif grid[r, c] == 2: render_str += "A" # Agent
                elif grid[r, c] == 3: render_str += "G" # Goal
            render_str += "\n"
        return render_str

    def _render_rgb(self):
        """Renders the environment state to an RGB array using Matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not installed, cannot render RGB array.")
            return None
        if self._maze is None: return None

        if self.window is None:
            plt.ion() # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.window = self.fig # Use fig as window reference
        else:
            self.ax.clear()

        self.ax.set_xlim(-0.5, self.grid_width - 0.5)
        self.ax.set_ylim(-0.5, self.grid_height - 0.5)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.invert_yaxis() # Match grid coordinates
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Draw maze walls
        wall_color = 'black'
        path_color = 'white'
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                color = wall_color if self._maze[r, c] == 1 else path_color
                rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=0, facecolor=color)
                self.ax.add_patch(rect)

        # Draw goal
        goal_r, goal_c = self._target_location
        goal_patch = patches.Rectangle((goal_c - 0.4, goal_r - 0.4), 0.8, 0.8, linewidth=0, facecolor='lime', alpha=0.7)
        self.ax.add_patch(goal_patch)

        # Draw agent
        agent_r, agent_c = self._agent_location
        agent_patch = patches.Circle((agent_c, agent_r), radius=0.35, facecolor='dodgerblue')
        self.ax.add_patch(agent_patch)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Convert canvas to numpy array
        img_buf = self.fig.canvas.buffer_rgba()
        img = np.frombuffer(img_buf, dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
        # Return RGB (drop alpha channel)
        return img[:, :, :3]

    def close(self):
        if self.window is not None and MATPLOTLIB_AVAILABLE:
            plt.close(self.fig)
            self.window = None
            plt.ioff() # Turn off interactive mode