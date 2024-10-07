import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionNet(nn.Module):
    def __init__(self, state_dim, constraint_dim, attention_dim=64):
        super(AttentionNet, self).__init__()

        # Linear layers to project the state and constraints into query, key, and value vectors
        self.query_layer = nn.Linear(state_dim, attention_dim)
        self.key_layer = nn.Linear(constraint_dim, attention_dim)
        self.value_layer = nn.Linear(state_dim, attention_dim)

        # Output linear layer after applying attention
        self.output_layer = nn.Linear(attention_dim, state_dim)

    def forward(self, state, constraints):
        # Project state and constraints into queries, keys, and values
        query = self.query_layer(state).unsqueeze(0)  # Ensure query is 2D
        key = self.key_layer(constraints).unsqueeze(0)  # Ensure key is 2D
        value = self.value_layer(state).unsqueeze(0)  # Ensure value is 2D

        # Calculate attention scores (similarity between query and key)
        attention_scores = torch.matmul(query, key.T) / (key.size(-1) ** 0.5)  # Scaled dot-product
        attention_weights = F.softmax(attention_scores, dim=-1)  # Softmax to get weights

        # Compute weighted sum of values
        attention_output = torch.matmul(attention_weights, value)

        # Pass through the final output layer
        output = self.output_layer(attention_output.squeeze(0))  # Remove the extra dimension

        return output