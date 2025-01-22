import torch
from torch import nn


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
