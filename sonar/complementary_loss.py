import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from comp_guided_conv import *
import warnings
from torch.autograd import Variable

# Ignore all warnings
warnings.filterwarnings("ignore")



class ComplementaryLoss(nn.Module):
    def __init__(self,guided_model,weight=0.1,device = 'cuda:7') -> None:
        super().__init__()
        self.weight = weight
        self.guided_model = guided_model.module


    def layer_loss(self,layer):
        metric = nn.MSELoss()
        x = layer.input_feature_map.clone()
        x_recon = layer.ForwardForLoss()

        # Calculate the difference in both height and width
        height_diff = x_recon.shape[-2] - x.shape[-2]
        width_diff = x_recon.shape[-1] - x.shape[-1]
        # Pad the original feature map or reconstructed map accordingly
        if height_diff > 0 or width_diff > 0:
            padding = (0, width_diff, 0, height_diff)
            x = F.pad(x, padding, mode='constant', value=0)
        else:
            padding = (0, -width_diff, 0, -height_diff)
            x_recon = F.pad(x_recon, padding, mode='constant', value=0)

        return metric(x,x_recon)

    def forward(self):
        loss = torch.tensor(0.0, requires_grad=True)

        # Iterate through the top-level modules of guided_model
        for name, layer in self.guided_model.guided_model.named_children():
            # If the layer is GuidedConv, apply the loss
            if isinstance(layer, GuidedConv):
                loss = loss + self.layer_loss(layer)

            # If the layer is a Sequential (or another container), iterate through the submodules
            if isinstance(layer, torch.nn.Sequential):
                for subname, sublayer in layer.named_children():
                    # If the sub-layer is GuidedConv, apply the loss
                    if isinstance(sublayer, GuidedConv):
                        loss = loss + self.layer_loss(sublayer)
                    else:

                    # If it contains more modules, iterate again for deeper layers
                        for subsubname, sub_sublayer in sublayer.named_children():
                            if isinstance(sub_sublayer, GuidedConv):
                                loss = loss + self.layer_loss(sub_sublayer)

        return loss*0.01


import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
class CombinedLoss(nn.Module):
    def __init__(self, lambda_reg=1e-2, weight=None):
        """
        Combines weighted cross-entropy loss with low-pass regularization for a specific layer.
        Args:
            lambda_reg (float): Weight for the regularization term.
            weight (torch.Tensor, optional): Weight for the classes in cross-entropy loss.
        """
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)  # Weighted cross-entropy
        self.lambda_reg = lambda_reg

    def forward(self, predictions, targets, model):
        """
        Forward pass for the combined loss.
        Args:
            predictions (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground-truth labels.
            model (nn.Module): The model being trained.
        Returns:
            torch.Tensor: Combined loss value.
        """
        # Compute weighted cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets)

        # Access the specific convolutional layer's kernel
        kernel = model.module.content_module.content_conv.weight  # Shape: [out, in, kernel_size]

        # Compute low-pass regularization loss for the kernel
        reg_loss = self._low_pass_regularizer(kernel)

        # Total loss
        total_loss = ce_loss + self.lambda_reg * reg_loss
        return total_loss

    def _low_pass_regularizer(self, kernel):
        """
        Low-pass regularizer combining entropy maximization and variance minimization.
        Args:
            kernel: Convolutional kernel of shape [out, in, kernel_size].
        Returns:
            torch.Tensor: Regularization term for the kernel.
        """
        eps = 1e-8
        # Use the absolute value of the kernel
        kernel = torch.abs(kernel.clone())

        # Normalize along the kernel size dimension
        kernel_sum = kernel.sum(dim=-1, keepdim=True) + eps
        normalized_kernel = kernel / kernel_sum

        # Entropy regularization
        entropy = -normalized_kernel * torch.log(normalized_kernel + eps)
        entropy = entropy.sum(dim=-1).mean()  # Average entropy across filters

        # Variance regularization
        mean = kernel.mean(dim=-1, keepdim=True)
        variance = ((kernel - mean) ** 2).mean(dim=-1).mean()  # Average variance

        # Combine regularization terms
        lambda_H = 1e-2  # Weight for entropy
        lambda_var = 1e-2  # Weight for variance
        return -lambda_H * entropy + lambda_var * variance
