import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedConv1dWithResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same',padding_mode='zeros', reduction_channels=1):
        super().__init__()
        self.constrained_conv = ConstrainedConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride= stride,padding= padding,padding_mode=padding_mode,bias=None)
        self.reduce_dim = nn.Conv1d(in_channels=out_channels, out_channels=reduction_channels, kernel_size=1, stride=1,padding_mode=padding_mode,padding=padding,bias=None)

    def forward(self, x):
        # Apply constrained convolution
        constrained_output = self.constrained_conv(x)
        # Reduce dimension back to 1 channel
        reduced_output = self.reduce_dim(constrained_output)
        return reduced_output


    def apply_constraints(self):
        # Apply constraints to the constrained_conv
        self.constrained_conv.apply_constraints()


class ConstrainedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # Apply the constraint: Subtract mean along filter dimensions
        constrained_weights = self.weight - self.weight.sum(dim=(2), keepdim=True)
        return F.conv1d(x, constrained_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def apply_constraints(self):
        # Set w_k(0) = 0 for all filters
        self.weight.data[:, :,2] = 0
        
        # Normalize weights so that sum(w_k) = 1 for all filters
        norm = self.weight.data.sum(dim=2, keepdim=True)
        self.weight.data = self.weight.data / (norm + 1e-8)  # Avoid division by zero
        
        # Optionally set specific values (e.g., -1 for w_k(0))
        self.weight.data[:, :, 2] = -1

