"""SONAR-Lite: dual-encoder + small MLP classifier (paper Sec. 4.3)."""

import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import argparse

from sonar.srm_filters import ConstrainedConv1dWithResidual
# SSLModel reused from the dual-encoder definition in guided_model.py
from sonar.guided_model import SSLModel


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        self.fc = nn.Linear(embed_dim*2, embed_dim)  # Optional FC layer

    def forward(self, x_low, x_high):
        # Cross-Attention
        attn_output_1, _ = self.attention(x_low, x_high, x_high)  # F1 queries F2
        attn_output_2, _  = self.attention(x_high, x_low, x_low)  # F2 queries F1
        ## output shape: 2 features of [B,201,1024]
        combined = torch.cat([attn_output_1, attn_output_2], dim=-1)  # [B,201, 2048]
        # Optional FC layer for further fusion
        fused = self.fc(combined)  # [B, 201, 1024]   
        return fused

class SmallClassifier(nn.Module):
    def __init__(self, emb_dim=1024, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//4)  # for binary classification
        self.fc3 = nn.Linear(hidden_dim//4,2)  # for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x





class GuidedModel(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        
        self.srm_module = ConstrainedConv1dWithResidual(1,30,5)

        #### 
        # create content network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        #self.sls_low  =SLSLayer(self.device)
        #### 
        # create noise network wav2vec 2.0
        ####
        self.noise_ssl_model = SSLModel(self.device)
        self.noise_ssl_model.requires_grad=True
        #self.sls_high = SLSLayer(self.device)
        #self.merge_outputs = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

        self.MLP_cls = SmallClassifier()
        self.attention = CrossAttentionFusion()

    def forward(self, x):
        x_copy = x.clone()
        
        
        x = self.srm_module(x)
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_content_ssl_feat = self.ssl_model.extract_feat(x_copy.squeeze(1))
        x_noise_ssl_feat = self.noise_ssl_model.extract_feat(x.squeeze(1))
        x = self.attention(x_content_ssl_feat,x_noise_ssl_feat)

        x = self.MLP_cls(x)
        
        return x
        
        


