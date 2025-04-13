# environments/algo_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RepeatCopyEnv(gym.Env):
    """
    Algorithmic Environment: Repeat Copy Task.
    The agent sees a sequence of binary vectors followed by a delimiter.
    Then, it sees a repeat count R.
    The agent must output the initial sequence R times.

    Observation: [vector_element, start_flag, end_flag, repeat_flag]
    vector_element: 0 or 1 (or -1 if no vector element is presented)
    start_flag: 1 if this is the start of the input sequence, 0 otherwise.
    end_flag: 1 if this is the delimiter after the input sequence, 0 otherwise.
    repeat_flag: 1 if the repeat count R is presented now, 0 otherwise.

    Action: Output 0 or 1.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, max_seq_len=5, max_repeats=3, render_mode=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_repeats = max_repeats

        # Observation space: [element (-1,0,1), start(0,1), end(0,1), repeat(0,1)]
        # Using Box for simplicity, although could be MultiDiscrete.
        # Range for element needs adjustment if using Box. Let's map -1 to 0, 0 to 1, 1 to 2.
        self.obs_dim = 4
        self.obs_size = self.obs_dim # Provide obs_size attribute
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([2, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Action space: Output 0 or 1
        self.action_space = spaces.Discrete(2)

        self._sequence = None
        self._repeat_count = 0
        self._current_step = 0
        self._total_input_steps = 0
        self._total_output_steps = 0
        self._agent_output = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _generate_sequence(self):
        seq_len = self.np_random.integers(1, self.max_seq_len + 1)
        self._sequence = self.np_random.integers(0, 2, size=seq_len).tolist()
        self._repeat_count = self.np_random.integers(1, self.max_repeats + 1)

        # Calculate total steps required
        self._total_input_steps = seq_len + 1 + 1 # Sequence + Delimiter + Repeat Count
        self._total_output_steps = seq_len * self._repeat_count
        self._max_steps = self._total_input_steps + self._total_output_steps

    def _get_obs(self):
        element = 0 # Mapped -1 (no element)
        start_flag = 0
        end_flag = 0
        repeat_flag = 0

        if self._current_step == 0: # Start of sequence
            start_flag = 1
            element = self._sequence[0] + 1 # Map 0/1 to 1/2
        elif self._current_step < len(self._sequence): # Middle of sequence
            element = self._sequence[self._current_step] + 1
        elif self._current_step == len(self._sequence): # Delimiter after sequence
            end_flag = 1
        elif self._current_step == len(self._sequence) + 1: # Present repeat count
            repeat_flag = 1
            # Encode repeat count simply by setting element to repeat_count + 1? Max val needs check.
            # Let's keep element = 0 here, agent must infer from repeat_flag=1.
            # Alternative: add repeat count directly to observation space? For now, keep it simple.
        else: # Output phase, no input element presented
            pass

        obs = np.array([element, start_flag, end_flag, repeat_flag], dtype=np.float32)
        return obs

    def _get_info(self):
        return {
            "current_step": self._current_step,
            "input_sequence": self._sequence,
            "repeat_count": self._repeat_count,
            "output_phase": self._current_step >= self._total_input_steps
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_sequence()
        self._current_step = 0
        self._agent_output = []

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        if self._sequence is None:
             raise RuntimeError("Must call reset before step")

        terminated = False
        truncated = False
        reward = 0.0

        is_output_phase = self._current_step >= self._total_input_steps

        if is_output_phase:
            output_step_index = self._current_step - self._total_input_steps
            if output_step_index < self._total_output_steps:
                correct_output = self._sequence[output_step_index % len(self._sequence)]
                if action == correct_output:
                    reward = 1.0 / self._total_output_steps # Normalize total reward to ~1.0
                else:
                    reward = -1.0 / self._total_output_steps # Penalty for wrong output
                self._agent_output.append(action)
            else:
                # Agent acted after output should have finished
                truncated = True # Or should this be penalized? Let's truncate.

        # Advance time step
        self._current_step += 1

        # Check termination/truncation based on total steps
        if self._current_step >= self._max_steps:
            terminated = True # Task completed (or truncated if agent took too long)
            # Final check if output length matches expected length
            if len(self._agent_output) != self._total_output_steps:
                 pass # Already penalized during steps or truncated


        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()

    def _render_ansi(self):
         if self._sequence is None: return "Call reset()"
         obs = self._get_obs()
         obs_str = f"Step: {self._current_step}/{self._max_steps} Obs: [{int(obs[0])} E, {int(obs[1])} S, {int(obs[2])} D, {int(obs[3])} R]"
         state_str = f" Seq: {self._sequence} R: {self._repeat_count}"
         output_str = f" Agent Output: {self._agent_output}"
         return obs_str + state_str + output_str

    def close(self):
        pass
