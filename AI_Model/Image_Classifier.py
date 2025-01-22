import torch
from torch import nn
from torch.nn import MSELoss, L1Loss, HuberLoss


class SEBlock(nn.modules):
    def __init__(self, input_channel_dim, reduction_ratio, output_channel_dim):
        super(SEBlock, self).__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel_dim, input_channel_dim // reduction_ratio),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(input_channel_dim // reduction_ratio, input_channel_dim),
            nn.Sigmoid()
        )

        self.output_channel_dim = output_channel_dim

    def forward(self, x):
        return self.model(x)


class SE_ResNet(nn.Module):
    def __init__(self, input_channel_dim: int, output_channel_dim: int, downsampling: bool):
        super(SE_ResNet, self).__init__()

        self.downsampling = downsampling

        self.conv1 = nn.Conv2d(input_channel_dim, output_channel_dim,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            output_channel_dim, output_channel_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel_dim)
        self.se = SEBlock(output_channel_dim, 16, output_channel_dim)

        if downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channel_dim, output_channel_dim,
                          kernel_size=3, stride=2, bias=False, padding=1),
                nn.BatchNorm2d(output_channel_dim)
            )
        self.downsampling = downsampling

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsampling:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class FYP_CNN(nn.Module):
    def __init__(self, input_channel_dim, hidden_dim: int, input_numeric_dim: int, hidden_numeric_dim: int, combin_hidden_dim: int, num_output: int):
        super(FYP_CNN, self).__init__()
        self.ImagePreprocessing = nn.Sequential(
            SE_ResNet(input_channel_dim, input_channel_dim),
            SE_ResNet(input_channel_dim, hidden_dim, True),
            SE_ResNet(hidden_dim, hidden_dim, True),
        )

        self.NumericInformationPreprocessing = nn.Sequential(
            nn.Linear(input_numeric_dim, input_numeric_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_numeric_dim, hidden_numeric_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_numeric_dim, hidden_numeric_dim/2),
            nn.ReLU(inplace=True),
        )

        self.Combine = nn.Sequential(
            nn.Linear(64*64 + hidden_numeric_dim/2, combin_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(combin_hidden_dim, combin_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(combin_hidden_dim, num_output),
            nn.Sigmoid()
        )

    def forward(self, image, info):
        image_out = self.ImagePreprocessing(image)
        info_out = self.NumericInformationPreprocessing(info)
        combined = torch.cat((image_out, info_out), 1)
        return self.Combine(combined)


class LossFunction(nn.Module):
    def __init__(self, importance: list = [1, 1, 1, 1, 1, 1, 1, 1]):
        super(LossFunction, self).__init__()
        importance = torch.tensor(importance)
        self.MSE_Mask = torch.tensor([1, 0, 0, 1, 1, 1, 0, 0]) * importance
        self.MAE_Mask = torch.tensor([0, 1, 0, 0, 0, 0, 1, 1]) * importance
        self.Huber_Mask = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0]) * importance
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
        return self._MSE(output, target) + self._MAE(output, target) + self._Huber(output, target)
