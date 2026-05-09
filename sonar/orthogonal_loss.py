


import torch
import torch.nn.functional as F

def circular_convolution_direct(w1, w2):
    """
    Perform circular convolution between two 1D filters.
    Both filters are assumed to have the same kernel size.
    """
    kernel_size = w1.shape[-1]
    # Circular padding for kernel 1
    padded_w1 = torch.cat([w1, w1[..., :-1]], dim=-1)  # Wrap around
    
    # Convolution using `F.conv1d`
    conv_result = F.conv1d(
        padded_w1.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
        w2.unsqueeze(0).unsqueeze(0),        # Add batch and channel dims
        padding=0                            # No additional padding
    ).squeeze()                              # Remove batch and channel dims
    return conv_result

def antisymmetry_loss(filters):
    """
    Compute the antisymmetry loss for a batch of filters.
    Args:
        filters: Tensor of shape [num_of_filters, kernel_size]
    Returns:
        Frobenius norm loss enforcing antisymmetry for all pairs of filters.
    """
    num_filters, kernel_size = filters.shape
    loss = 0.0
    
    # Iterate through all unique pairs of filters
    for i in range(num_filters):
        for j in range(i + 1, num_filters):  # Avoid duplicate and self-pairs
            w1, w2 = filters[i], filters[j]
            
            # Transpose (flip) the kernels
            w1_t = torch.flip(w1, dims=[-1])
            w2_t = torch.flip(w2, dims=[-1])
            
            # Circular convolution
            conv1 = circular_convolution_direct(w1_t, w2)
            conv2 = circular_convolution_direct(w2_t, w1)
            
            # Compute antisymmetric term
            antisymmetric_kernel = conv1 + conv2
            
            # Add Frobenius norm of the antisymmetric term
            loss += torch.norm(antisymmetric_kernel, p='fro')**2
    return loss



import torch
import torch.nn as nn
import torch.nn.functional as F

class HellingerLoss(nn.Module):
    """
    Custom loss that encourages:
    - Hellinger distance = 0 if label == 1
    - Hellinger distance = 1 if label == 0
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x_low, x_high, labels):
        """
        Args:
            x_low:   [B, 201, 1024]
            x_high:  [B, 201, 1024]
            labels:  [B,] with values in {0, 1}
            
        Returns:
            A scalar loss enforcing H=0 for label=1 and H=1 for label=0.
        """
        # 1) Convert logits to probability distributions along dim=-1
        p_low  = F.softmax(x_low,  dim=-1)  # [B, 201, 1024]
        p_high = F.softmax(x_high, dim=-1)  # [B, 201, 1024]
        
        # 2) Compute the element-wise sqrt difference
        #    shape => [B, 201, 1024]
        sqrt_diff = torch.sqrt(p_low) - torch.sqrt(p_high)
        
        # 3) Compute the Hellinger distance per (b, i)
        #    H(b, i) = 1/sqrt(2) * sqrt( sum_j( (sqrt(p_low) - sqrt(p_high))^2 ) )
        #    shape => [B, 201]
        hellinger = (1.0 / (2.0**0.5)) * torch.norm(sqrt_diff, p=2, dim=-1)
        
        # 4) Build the target for each sample in the batch, broadcast to 201
        #    if labels[b] = 1 => target=0
        #    if labels[b] = 0 => target=1
        #    shape => [B, 201]
        targets = (1 - labels).unsqueeze(-1).float().expand_as(hellinger)
        
        # 5) Compute MSE between Hellinger distances and targets
        loss = torch.mean((hellinger - targets)**2)
        
        return loss
