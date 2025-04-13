import torch
from torch import nn
from torch.distributions import Categorical
from typing import Dict, Tuple

class LSTMBaseline(nn.Module):
    """Standard LSTM baseline model with the same interface as DNC."""
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_actions: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        # LSTM Cell
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        # Output heads
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1) # For Actor-Critic / PPO

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Simple initialization (can be customized)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                for name, param in module.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                        # Initialize forget gate bias to 1
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        nn.init.constant_(param[start:end], 1.)
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param)


    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initializes the LSTM state dictionary."""
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return {"controller": (h, c)}

    def forward(self,
                x: torch.Tensor,
                prev_state: Dict[str, torch.Tensor]
               ) -> Tuple[Categorical, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs one step of LSTM computation.

        Args:
            x: Input tensor (batch_size, input_size)
            prev_state: Dictionary containing the previous LSTM state {"controller": (h, c)}.

        Returns:
            action_dist: Categorical distribution over actions.
            value: Estimated state value.
            new_state: Dictionary containing the updated LSTM state.
        """
        device = x.device
        x = x.to(device)

        prev_h, prev_c = prev_state["controller"]
        prev_h, prev_c = prev_h.to(device), prev_c.to(device)

        # LSTM step
        h, c = self.lstm_cell(x, (prev_h, prev_c))

        # Output - Policy and Value
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1) # Ensure value is (B,)
        action_dist = Categorical(logits=logits)

        # Assemble new state dictionary
        new_state = {"controller": (h, c)}

        return action_dist, value, new_state
