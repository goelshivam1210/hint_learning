import torch.nn as nn
import numpy as np
import torch

class AttentionNet(nn.Module):
    def __init__(self, obs_space_shape, constraint_dim, hidden_dim=64):
        super(AttentionNet, self).__init__()
        
        # Flatten the observation space shape
        input_dim = np.prod(obs_space_shape)  # Flattened shape to an integer

        # Attention layers
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.query_layer = nn.Linear(constraint_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

        # Set the output_dim to be used by the Tianshou framework
        self.output_dim = hidden_dim  # The final output dimensionality

    def forward(self, obs, constraints):

        print (f"observation = {obs}")
        print (f"observation-shape = {obs.shape}")
        print(f"constraints received: {constraints}")  
        # Ensure the inputs are Tensors
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.key_layer.weight.device)
        if isinstance(constraints, np.ndarray):
            constraints = torch.tensor(constraints, dtype=torch.float32, device=self.key_layer.weight.device)
    

        # Compute attention mechanism
        keys = self.key_layer(obs)
        queries = self.query_layer(constraints)
        values = self.value_layer(obs)

        # Attention mechanism
        attention_scores = torch.matmul(queries, keys.T) / np.sqrt(keys.size(-1))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)

        # Pass through the output layer
        output = self.output_layer(attention_output)
        return output