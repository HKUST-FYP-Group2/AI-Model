import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, channel_dim, patchSize, embedding_dim):
        super(PatchEmbedding, self).__init__()
        self.channel_dim = channel_dim
        self.patchSize = patchSize
        self.embedding_dim = embedding_dim

        self.embedder = nn.Linear(channel_dim * patchSize * patchSize, embedding_dim)
    
    def forward(self, x):
        transposedX = x.permute(0, 2, 3, 1)
        currentShape = transposedX.shape
        transformedX = torch.reshape(transposedX, 
            (currentShape[0], 
             currentShape[1]//self.patchSize, self.patchSize, 
             currentShape[2]//self.patchSize, self.patchSize, 
             self.channel_dim))
        transformedX = transformedX.permute(0, 1, 3, 2, 4, 5)
        currentShape = transformedX.shape
        transformedX = torch.reshape(transformedX, (currentShape[0], currentShape[1]*currentShape[2], -1))
        return self.embedder(transformedX)