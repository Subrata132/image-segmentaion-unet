import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
            self,
            input_channel,
            output_channel,
            kernel=3,
            stride=1,
            padding=1
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channel, retrain=True):
        super().__init__()
        self.conv1 = ConvBlock(input_channel=input_channel, output_channel=32)
        self.conv2 = ConvBlock(input_channel=32, output_channel=64)
        self.conv3 = ConvBlock(input_channel=64, output_channel=128)
        self.conv4 = ConvBlock(input_channel=128, output_channel=256)
        
