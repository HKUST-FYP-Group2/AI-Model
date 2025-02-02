from torch import nn
import torch

class ProcessAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super(ProcessAttention, self).__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.attn = nn.MultiheadAttention(inner_dim, heads, batch_first=True)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, X):

        q: torch.Tensor = self.to_q(X) # (batch_size, num_patches, inner_dim)
        k: torch.Tensor = self.to_k(X) # (batch_size, num_patches, inner_dim)
        v: torch.Tensor = self.to_v(X) # (batch_size, num_patches, inner_dim)
        
        attn_output, _ = self.attn(q, k, v) # (batch_size, num_patches, inner_dim)
        return self.to_out(attn_output) # (batch_size, num_patches, dim)
    
class ProcessAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super(ProcessAttentionBlock, self).__init__()
        
        self.attn = ProcessAttention(dim = dim, heads = heads, dim_head = dim_head)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x