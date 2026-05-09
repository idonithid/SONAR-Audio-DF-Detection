import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sonar.orthogonal_loss import *

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





# Normalize kernel eigenvalues
def normalize_kernel(kernel, N=10000):
    # Zero-pad the kernel to length N
       # Ensure kernel is a 1D numpy array
    kernel = kernel.view(-1) # Convert to numpy array for easier manipulation

    # Initialize zero-padded kernel
    kernel_padded = torch.zeros(N, dtype=torch.float32)
    kernel_padded[:len(kernel)] = kernel  # Assign kernel values to the start
    
    W = torch.fft.fft(kernel_padded)

    # Find the maximum eigenvalue (largest magnitude)
    lambda_max = torch.max(torch.abs(W))
    
    # Normalize the FFT of the kernel
    kernel_normalized = kernel / lambda_max
    return kernel_normalized



#Define Polynomial Convolutional Transformation with Spectrum Loss
class PolynomialConv1D(nn.Module):
    def __init__(self,kernel_size = 7,th_low=0.7,th_high=1,max_freq=10000,device = "cuda:0"):
        super(PolynomialConv1D, self).__init__()
        # Initialize Conv1D layer
        self.base_conv= nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, bias=False,padding='same',padding_mode='circular').to(device)

        with torch.no_grad():  # Ensure no gradients are tracked
            normalized_weights= normalize_kernel(self.base_conv.weight)
            self.base_conv.weight.copy_(torch.tensor(normalized_weights)) # Update the layer's weights

        for param in self.base_conv.parameters():
            param.requires_grad = False
            # Zero-pad the kernel to the desired length
        padded_kernel = torch.zeros(max_freq)
        padded_kernel[:kernel_size] = self.base_conv.weight.data.view(-1).clone()
        
        # Compute the FFT of the padded kernel
        fft_result = torch.fft.fft(padded_kernel)
        
        # Compute the frequency axis
        freqs = torch.fft.fftfreq(max_freq, d=1/2*max_freq)
        positive_freqs = freqs[:max_freq // 2]  # Take only positive frequencies
        magnitude = torch.abs(fft_result[:max_freq // 2])  # Magnitudes of the positive frequencies
    
        lambdas_init = magnitude
        print(f"Initial Eigenvalues: {lambdas_init}")


        self.lambdas_init = lambdas_init.to(device)

        # Learnable parameters for the polynomial
        self.a = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)
        self.b = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)
        self.c = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)
        # Thresholds for eigenvalues
        self.th_low = th_low
        self.th_high = th_high
       

    def forward(self, x):
        # Compute convolutions
        w1 = self.base_conv(x)  # w
        w2 = self.base_conv(w1.clone())  # w^2
        w3 = self.base_conv(w2.clone())  # w^3
        # Polynomial combination
        output = self.a * w1 + self.b * w2 + self.c * w3
        return output


    def compute_eigenvalues(self):
            # Compute adjusted eigenvalues using the polynomial parameters
            return self.a * self.lambdas_init + self.b * (self.lambdas_init**2) + self.c * (self.lambdas_init**3)

    def spectrum_loss(self, lambdas):
        """
        Compute the area + threshold ReLU loss for the eigenvalues.
        :param lambdas: Tensor of eigenvalues (1D tensor)
        :return: Loss value
        """
        # ReLU threshold loss
        loss_low = torch.relu(self.th_low - lambdas).sum()
        loss_high = torch.relu(lambdas - self.th_high).sum()
        return loss_low + loss_high 


import torch
import torch.nn as nn

class MultiPolynomialConv1D(nn.Module):
    def __init__(self, num_of_filters, kernel_size=7, th_low=0.7, th_high=1, max_freq=10000,device='cuda:0'):
        super(MultiPolynomialConv1D, self).__init__()
        self.filters = nn.ModuleList([
            PolynomialConv1D(kernel_size=kernel_size, th_low=th_low, th_high=th_high, max_freq=max_freq,device=device)
            for _ in range(num_of_filters)
        ])
        self.filter_weights = torch.stack([
            self.filters[i].base_conv.weight.view(-1)  # Flatten weights of each filter
            for i in range(num_of_filters)
        ], dim=0)  # Stack along the first dimension
    def forward(self, x):
        """
        Forward pass for all filters.
        :param x: Input tensor of shape (batch_size, 1, sequence_length)
        :return: List of outputs, one for each filter
        """
        outputs = [filter_module(x) for filter_module in self.filters]
        return torch.stack(outputs, dim=1)  # Stack along filter dimension for easier handling
    
    def compute_all_eigenvalues(self):
        """
        Compute eigenvalues for all filters.
        :return: List of eigenvalue tensors, one for each filter
        """
        all_eigenvalues = [filter_module.compute_eigenvalues() for filter_module in self.filters]
        return torch.stack(all_eigenvalues, dim=0)  # Stack along filter dimension

    def spectrum_loss(self):
        """
        Compute combined spectrum loss across all filters.
        :return: Total loss for all filters
        """
        total_loss = 0
        for filter_module in self.filters:
            lambdas = filter_module.compute_eigenvalues()
            total_loss += filter_module.spectrum_loss(lambdas)
        return total_loss

    def loss(self):
        spectrum_loss = self.spectrum_loss()
        ortho_loss = antisymmetry_loss((self.filter_weights))
        return 0.000001*spectrum_loss  + 0.00001*ortho_loss