from torch import nn

class DecodeBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, output_embedding_dim):
        super(DecodeBlock, self).__init__()
        
        self.decode_attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, output_embedding_dim)
        )
        
        self.norm1 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, latentArray):
        x, _ = self.decode_attn(x, latentArray, latentArray)
        return self.decoder(self.norm1(x))