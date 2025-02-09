import torch
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, input_channel_dim, reduction_ratio, output_channel_dim):
        super(SEBlock, self).__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # (B, input_channel_dim, H, W) -> (B, input_channel_dim, 1, 1)
            nn.Conv2d(input_channel_dim, input_channel_dim // reduction_ratio, kernel_size=1), 
            # (B, input_channel_dim, 1, 1) -> (B, input_channel_dim // reduction_ratio, 1, 1)
            
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(input_channel_dim // reduction_ratio, output_channel_dim, kernel_size=1),
            # (B, input_channel_dim // reduction_ratio, 1, 1) -> (B, output_channel_dim, 1, 1)
            
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class SE_ResNet(nn.Module):
    def __init__(self, input_channel_dim: int, output_channel_dim: int, reduction_ratio:int, downsampling: bool, imgSize:tuple[int]):
        super(SE_ResNet, self).__init__()

        self.downsampling = downsampling
        self.imgSize = imgSize

        self.conv1 = nn.Conv2d(input_channel_dim, input_channel_dim*2,
                               kernel_size=3, stride=1, padding=1) # (B, input_channel_dim, H, W) -> (B, input_channel_dim*2, H, W)
        self.bn1 = nn.BatchNorm2d(input_channel_dim*2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = None
        if input_channel_dim != output_channel_dim:
            if downsampling:
                self.conv2 = nn.Conv2d(input_channel_dim*2, output_channel_dim,
                                                kernel_size=1, stride=2, bias=False) 
                # (B, input_channel_dim*2, H, W) -> (B, output_channel_dim, H/2, W/2)
            else:
                self.conv2 = nn.Conv2d(input_channel_dim*2, output_channel_dim,
                                                kernel_size=1, stride=1, bias=False)
                # (B, input_channel_dim*2, H, W) -> (B, output_channel_dim, H, W)
        self.bn2 = nn.BatchNorm2d(num_features=output_channel_dim) 
        
        
        self.se = SEBlock(input_channel_dim, reduction_ratio,
                          input_channel_dim*2)
        # (B, input_channel_dim, H, W) -> (B, output_channel_dim, 1, 1)
        self.scaler = None
        if downsampling:
            self.scaler = nn.ConvTranspose2d(input_channel_dim * 2, output_channel_dim,
                                            kernel_size=(imgSize[0]//2,imgSize[1]//2))
        else:
            self.scaler = nn.Conv2d(input_channel_dim * 2, output_channel_dim,
                                            kernel_size=imgSize)
        
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        residual = self.se(identity)
        residual = self.scaler(residual)
        
        return self.relu2(out + residual)


class SE_CNN(nn.Module):
    def __init__(self, input_channel_dim, hidden_dim: int, extractor_hidden_dim: int, num_output: int):
        super(SE_CNN, self).__init__()
        self.ImagePreprocessing = nn.Sequential(
            SE_ResNet(input_channel_dim, hidden_dim,
                      1, True,(256,256)),
            SE_ResNet(hidden_dim, hidden_dim*2,
                      hidden_dim//8, True, (128,128)),
            SE_ResNet(hidden_dim*2, hidden_dim,
                      hidden_dim//4,True, (64,64)),
        )

        self.extract = nn.Sequential(
            nn.Linear(32*32*hidden_dim, extractor_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(extractor_hidden_dim, num_output),
        )

    def forward(self, image):
        image_out = self.ImagePreprocessing(image)
        image_out = torch.flatten(image_out, 1)
        return self.extract(image_out)
