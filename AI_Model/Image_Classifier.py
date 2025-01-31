import torch
from torch import nn
from torch.nn import MSELoss, L1Loss, HuberLoss


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


class FYP_CNN(nn.Module):
    def __init__(self, input_channel_dim, hidden_dim: int, input_numeric_dim: int, hidden_numeric_dim: int, combin_hidden_dim: int, num_output: int):
        super(FYP_CNN, self).__init__()
        self.ImagePreprocessing = nn.Sequential(
            SE_ResNet(input_channel_dim, hidden_dim,
                      1, True,(256,256)),
            SE_ResNet(hidden_dim, hidden_dim*2,
                      hidden_dim//8, True, (128,128)),
            SE_ResNet(hidden_dim*2, hidden_dim,
                      hidden_dim//4,True, (64,64)),
        )

        self.NumericInformationPreprocessing = nn.Sequential(
            nn.Linear(input_numeric_dim, hidden_numeric_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_numeric_dim*2, hidden_numeric_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_numeric_dim, hidden_numeric_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.Combine = nn.Sequential(
            nn.Linear(32*32*hidden_dim + hidden_numeric_dim//2, combin_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(combin_hidden_dim, num_output),
            nn.Sigmoid()
        )

    def forward(self, image, info):
        image_out = self.ImagePreprocessing(image)
        info_out = self.NumericInformationPreprocessing(info)
        image_out = torch.flatten(image_out, 1)
        combined = torch.cat((image_out, info_out), 1)
        return self.Combine(combined)


class LossFunction(nn.Module):
    def __init__(self, importance: list = [1, 1, 1, 1, 1, 1, 1, 1], device:torch.device = torch.device("cpu")):
        super(LossFunction, self).__init__()
        importance = torch.tensor(importance, device=device)
        self.MSE_Mask = torch.tensor([1, 0, 0, 1, 1, 1, 0, 0],device=device) * importance
        self.MAE_Mask = torch.tensor([0, 1, 0, 0, 0, 0, 1, 1],device=device) * importance
        self.Huber_Mask = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], device=device) * importance
        self._MSEFunction = MSELoss()
        self._MAEFunction = L1Loss()
        self._HuberFunction = HuberLoss()

    def _MSE(self, output, target):
        return self._MSEFunction(output, target) * self.MSE_Mask

    def _MAE(self, output, target):
        return self._MAEFunction(output, target) * self.MAE_Mask

    def _Huber(self, output, target):
        return self._HuberFunction(output, target) * self.Huber_Mask

    def forward(self, output, target):
        return torch.mean(self._MSE(output, target) + self._MAE(output, target) + self._Huber(output, target))
