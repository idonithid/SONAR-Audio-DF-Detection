import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyRegularizer(nn.Module):
    def __init__(self, model,device):
        super(EntropyRegularizer, self).__init__()
        self.model = model
        self.device = device
    def _calculate_entropy(self, weights):
        # Normalize weights to form a probability distribution
        abs_weights = weights.abs()
        prob_dist = abs_weights / abs_weights.sum()  # Normalize to sum to 1

        # Calculate Shannon entropy
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))  # Small epsilon to avoid log(0)
        return entropy

    def _fc_layer_entropy(self, weights):
        # Calculate entropy row-wise for fully connected (FC) layers
        row_entropy = torch.tensor([self._calculate_entropy(row) for row in weights])
        return row_entropy.sum()  # Sum entropies of all rows

    def forward(self):
        total_entropy = torch.tensor(0.0)
        total_entropy = total_entropy.to(self.device)
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):  # For FC layers
                total_entropy += self._fc_layer_entropy(layer.weight)
            elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):  # For Conv layers
                layer_entropy = self._calculate_entropy(layer.weight)
                layer_entropy = layer_entropy.to(self.device)
                total_entropy += layer_entropy
        return total_entropy