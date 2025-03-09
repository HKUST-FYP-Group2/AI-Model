from torch import nn
from .Embedders.Image_Embedder import PatchEmbedding
from .Transformer_Blocks.Process_Blocks import ProcessAttentionBlock
from .Transformer_Blocks.Decode_Blocks import DecodeBlock
        

class OutputConverter(nn.Module):
    def __init__(self, embedding_dim, input_feature_dim, num_classes):
        super(OutputConverter, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim*input_feature_dim, num_classes),
            nn.Linear(num_classes, num_classes*embedding_dim)
        )
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        flattenedX = x.flatten(1)
        output = self.model(flattenedX)
        return output.view(-1, self.num_classes, self.embedding_dim)

class PerceiverIO(nn.Module):
    def __init__(self,embedding_dim, 
                 input_feature_dim ,num_classes, 
                 num_layers, num_heads, 
                 mlp_dim):
        super(PerceiverIO, self).__init__()
        self.image_embedder = PatchEmbedding(3, 16, embedding_dim)

        self.latentFeatures = nn.Linear(embedding_dim, embedding_dim)
        self.outputFeatures = OutputConverter(embedding_dim, input_feature_dim, num_classes)
        
        self.encoder = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        
        self.encoder_blocks =nn.Sequential(*[
            ProcessAttentionBlock(dim = embedding_dim, heads = num_heads, dim_head = embedding_dim // num_heads, mlp_dim = mlp_dim)
          for _ in range(num_layers)
        ])

        self.decoder = DecodeBlock(embedding_dim, num_heads, 
                                   1)
        

    def forward(self, img):
        img_embedding = self.image_embedder(img)
        
        latentArray = self.latentFeatures(img_embedding)
        
        latentArrays,_ = self.encoder(latentArray, img_embedding, img_embedding)
        beforeDecode = self.encoder_blocks(latentArrays)
        
        outputArray = self.outputFeatures(beforeDecode)
        
        decodedOutput = self.decoder(outputArray, beforeDecode)
        decodedOutput = decodedOutput.squeeze(2)
        
        return decodedOutput
        
      


    