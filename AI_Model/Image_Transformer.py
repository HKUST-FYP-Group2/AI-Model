from torch import nn
import torch
from .Embedders.Image_Embedder import PatchEmbedding
from .Embedders.Numeric_Embedder import NumericEmedding
from .Transformer_Blocks.Process_Blocks import ProcessAttentionBlock
from .Transformer_Blocks.Decode_Blocks import DecodeBlock
        

class PerceiverIO(nn.Module):
    def __init__(self, embedding_dim, input_feature_dim ,ouput_feature_dim, num_layers, num_heads, mlp_dim):
        super(PerceiverIO, self).__init__()
        self.image_embedder = PatchEmbedding(3, 16, embedding_dim)
        self.numeric_embedder = NumericEmedding(1, embedding_dim)
        
        self.latentFeatures = nn.Linear(embedding_dim, embedding_dim)
        self.outputFeatures = nn.Linear(input_feature_dim, ouput_feature_dim)
        
        self.encoder = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        
        self.encoder_blocks =nn.Sequential(*[
            ProcessAttentionBlock(dim = embedding_dim, heads = num_heads, dim_head = embedding_dim // num_heads, mlp_dim = mlp_dim)
          for _ in range(num_layers)
        ])

        self.decoder = DecodeBlock(embedding_dim, num_heads, 
                                   1)
        
    def outputConvert(self, x):
        transposedX = x.permute(0, 2, 1)
        output = self.outputFeatures(transposedX)
        return output.permute(0, 2, 1)
        

    def forward(self, img, X):
        img_embedding = self.image_embedder(img)
        numeric_embedding = self.numeric_embedder(X)
        
        combinedInput = torch.cat((img_embedding, numeric_embedding), dim=1)
        latentArray = self.latentFeatures(combinedInput)
        
        latentArrays,_ = self.encoder(latentArray, combinedInput, combinedInput)
        beforeDecode = self.encoder_blocks(latentArrays)
        
        outputArray = self.outputConvert(combinedInput)
        
        decodedOutput = self.decoder(outputArray, beforeDecode)
        decodedOutput = decodedOutput.squeeze(2)
        
        return decodedOutput
        
      


    