'''
-----------------------------------------------------------------------
File: model.py
Creation Time: May 26th 2024, 1:05 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''

import math
import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt*(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, d_type = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (-math.log(10000.0) / d_model) ensure that the positional encoding for closer positions are more similar than for positions farther apart
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 0::1] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # Register the buffer so that it is not updated during the training process
        self.register_buffer('pe', pe)
    
    def forward(/self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # .requires_grad_(False) is a PyTorch method that's used to tell PyTorch not to compute gradients for this tensor during the backward pass. This is because the positional encoding is not a learnable parameter of the model, so we don't need to update it during training.
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.one(1)) # Multiplicative parameter
        self.bias = nn.Parameter(torch.zeros(1)) # Additive parameter
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias