# coding=UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnHead(nn.Module):
    """
    Graph Attention Head
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        in_drop: Input dropout rate
        coef_drop: Attention coefficient dropout rate
        activation: Activation function
        residual: Whether to use residual connection
    """
    def __init__(self, in_features, out_features, in_drop=0.0, coef_drop=0.0, 
                 activation=F.elu, residual=False):
        super(AttnHead, self).__init__()
        
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.activation = activation
        self.residual = residual
        
        # Learnable parameters
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_features, 1, kernel_size=1)
        self.conv3 = nn.Conv1d(out_features, 1, kernel_size=1)
        
        if residual and in_features != out_features:
            self.conv4 = nn.Conv1d(in_features, out_features, kernel_size=1)
        
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, seq, bias_mat, return_coef=False):
        """
        Forward pass
        
        Args:
            seq: Input sequence [batch_size, nb_nodes, in_features]
            bias_mat: Bias matrix for masking [batch_size, nb_nodes, nb_nodes]
            return_coef: Whether to return attention coefficients
        
        Returns:
            Output features [batch_size, nb_nodes, out_features]
            (Optional) Attention coefficients if return_coef=True
        """
        # Input dropout
        if self.in_drop > 0.0 and self.training:
            seq = F.dropout(seq, p=self.in_drop, training=True)
        
        # Transform: [B, N, F] -> [B, F, N] -> Conv1d -> [B, F', N] -> [B, N, F']
        seq_fts = self.conv1(seq.transpose(1, 2)).transpose(1, 2)
        
        # Attention mechanism
        f_1 = self.conv2(seq_fts.transpose(1, 2)).transpose(1, 2)  # [B, N, 1]
        f_2 = self.conv3(seq_fts.transpose(1, 2)).transpose(1, 2)  # [B, N, 1]
        
        logits = f_1 + f_2.transpose(1, 2)  # [B, N, N]
        logits = F.leaky_relu(logits)
        
        # Apply bias mask and softmax
        coefs = F.softmax(logits + bias_mat, dim=-1)  # [B, N, N]
        
        # Coefficient dropout
        if self.coef_drop > 0.0 and self.training:
            coefs = F.dropout(coefs, p=self.coef_drop, training=True)
        
        # Feature dropout
        if self.in_drop > 0.0 and self.training:
            seq_fts = F.dropout(seq_fts, p=self.in_drop, training=True)
        
        # Attention-weighted aggregation
        vals = torch.matmul(coefs, seq_fts)  # [B, N, F']
        
        # Batch normalization: [B, N, F'] -> [B, F', N] -> BN -> [B, N, F']
        ret = self.bn(vals.transpose(1, 2)).transpose(1, 2)
        
        # Residual connection
        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv4(seq.transpose(1, 2)).transpose(1, 2)
            else:
                ret = ret + seq
        
        # Activation
        ret = self.activation(ret)
        
        if return_coef:
            return ret, coefs
        else:
            return ret


def attn_head(seq, out_sz, bias_mat, activation=F.elu, in_drop=0.0, coef_drop=0.0, 
              residual=False, return_coef=False):
    """
    Functional interface for attention head (for backward compatibility)
    
    Note: This is a stateless version - in practice, use AttnHead class instead
    """
    # This would need to be instantiated properly in a module
    # For now, provide a warning
    raise NotImplementedError(
        "Use AttnHead class instead. Example:\n"
        "attn = AttnHead(in_features, out_sz, in_drop, coef_drop, activation, residual)\n"
        "output = attn(seq, bias_mat, return_coef)"
    )


class SimpleAttLayer(nn.Module):
    """
    Simple Attention Layer for sequence aggregation
    
    Args:
        hidden_size: Hidden size of input
        attention_size: Attention vector size
    """
    def __init__(self, hidden_size, attention_size):
        super(SimpleAttLayer, self).__init__()
        
        self.attention_size = attention_size
        self.hidden_size = hidden_size
        
        # Learnable parameters
        self.w_omega = nn.Parameter(torch.randn(hidden_size, attention_size) * 0.1)
        self.b_omega = nn.Parameter(torch.randn(attention_size) * 0.1)
        self.u_omega = nn.Parameter(torch.randn(attention_size) * 0.1)
    
    def forward(self, inputs, time_major=False, return_alphas=False):
        """
        Forward pass
        
        Args:
            inputs: Input tensor [batch_size, seq_len, hidden_size] or tuple of tensors
            time_major: Whether input is time major [seq_len, batch_size, hidden_size]
            return_alphas: Whether to return attention weights
        
        Returns:
            output: Aggregated output [batch_size, hidden_size]
            alphas: (Optional) Attention weights if return_alphas=True
        """
        # Handle Bi-RNN outputs (tuple of forward and backward)
        if isinstance(inputs, tuple):
            inputs = torch.cat(inputs, dim=2)
        
        # Convert to batch-first format
        if time_major:
            inputs = inputs.transpose(0, 1)  # [T, B, D] -> [B, T, D]
        
        # Compute attention scores
        # v = tanh(inputs * w_omega + b_omega)
        v = torch.tanh(torch.matmul(inputs, self.w_omega) + self.b_omega)  # [B, T, A]
        
        # vu = v * u_omega
        vu = torch.matmul(v, self.u_omega)  # [B, T]
        
        # Compute attention weights (per sequence)
        alphas = F.softmax(vu, dim=-1)  # [B, T]
        
        # Compute global attention weights (averaged over batch)
        myalphas = F.softmax(vu.mean(dim=0), dim=-1)  # [T]
        myalphas = myalphas.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        myalphas = myalphas.expand(inputs.size(0), -1, -1)  # [B, T, 1]
        
        # Weighted sum
        output = (inputs * myalphas).sum(dim=1)  # [B, D]
        
        myalphas = myalphas.squeeze(-1)  # [B, T]
        
        if return_alphas:
            return output, myalphas
        else:
            return output


class ConstantAttnHead(nn.Module):
    """
    Attention head with constant (adjacency-based) attention coefficients
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        in_drop: Input dropout rate
        coef_drop: Coefficient dropout rate
        activation: Activation function
        residual: Whether to use residual connection
    """
    def __init__(self, in_features, out_features, in_drop=0.0, coef_drop=0.0,
                 activation=F.elu, residual=False):
        super(ConstantAttnHead, self).__init__()
        
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.activation = activation
        self.residual = residual
        
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size=1, bias=False)
        
        if residual and in_features != out_features:
            self.conv_res = nn.Conv1d(in_features, out_features, kernel_size=1)
        
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, seq, bias_mat):
        """
        Forward pass with constant attention
        
        Args:
            seq: Input [batch_size, nb_nodes, in_features]
            bias_mat: Bias matrix [batch_size, nb_nodes, nb_nodes]
        
        Returns:
            Output [batch_size, nb_nodes, out_features]
        """
        # Compute adjacency matrix from bias
        adj_mat = 1.0 - bias_mat / (-1e9)
        
        # Input dropout
        if self.in_drop > 0.0 and self.training:
            seq = F.dropout(seq, p=self.in_drop, training=True)
        
        # Transform features
        seq_fts = self.conv1(seq.transpose(1, 2)).transpose(1, 2)
        
        # Use adjacency as attention
        logits = adj_mat
        coefs = F.softmax(F.leaky_relu(logits) + bias_mat, dim=-1)
        
        # Coefficient dropout
        if self.coef_drop > 0.0 and self.training:
            coefs = F.dropout(coefs, p=self.coef_drop, training=True)
        
        # Feature dropout
        if self.in_drop > 0.0 and self.training:
            seq_fts = F.dropout(seq_fts, p=self.in_drop, training=True)
        
        # Aggregation
        vals = torch.matmul(coefs, seq_fts)
        ret = self.bn(vals.transpose(1, 2)).transpose(1, 2)
        
        # Residual
        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_res(seq.transpose(1, 2)).transpose(1, 2)
            else:
                ret = ret + seq
        
        return self.activation(ret)


# Backward compatibility functions
def attn_head_const_1(seq, out_sz, bias_mat, activation=F.elu, in_drop=0.0, 
                      coef_drop=0.0, residual=False):
    """Functional interface for constant attention head"""
    raise NotImplementedError("Use ConstantAttnHead class instead")


def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, 
                 coef_drop=0.0, residual=False):
    """Sparse attention head - not implemented in PyTorch version"""
    raise NotImplementedError(
        "Sparse attention not directly supported in PyTorch. "
        "Use torch_sparse library or dense version instead."
    )
