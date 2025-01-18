import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionNet(nn.Module):
    def __init__(self, state_dim, constraint_dim, hidden_dim=64):
        super(AttentionNet, self).__init__()
        # Linear layers to learn transformations for Query, Key, Value
        self.query_layer = nn.Linear(constraint_dim, hidden_dim)
        self.key_layer = nn.Linear(state_dim, hidden_dim)
        self.value_layer = nn.Linear(state_dim, hidden_dim)

        # Output transformation
        self.output_layer = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, constraints):
        if state.dim() == 1:  # Add batch dimension if state is a single sample
            state = state.unsqueeze(0)
        if constraints.dim() == 1:  # Add batch dimension for constraints
            constraints = constraints.unsqueeze(0)

        # Transform constraints (query) and state (key, value)
        Q = self.query_layer(constraints)  # Query from constraints
        K = self.key_layer(state)         # Key from state
        V = self.value_layer(state)       # Value from state

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Compute weighted sum of values
        attention_output = torch.matmul(attention_weights, V)

        # Transform the output
        processed_state = self.output_layer(attention_output)
        return processed_state.squeeze(0) if state.size(0) == 1 else processed_state