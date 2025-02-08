from torch import nn

class NumericEmedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NumericEmedding, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.GELU(),
            nn.Linear(input_dim*2, output_dim),
            nn.GELU(),
        )
    
    def forward(self, x):
        x = x.unsqueeze(2)
        return self.model(x)
    
class OutputDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OutputDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.GELU(),
            nn.Linear(input_dim*2, output_dim),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.model(x)