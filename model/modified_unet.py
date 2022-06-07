import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transform
from . model import ConvBlock


class UpConvolution(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel=3,
            stride=2,
            padding=0,
            output_padding=1
    ):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

    def forward(self, x):
        return self.up_conv(x)


class ModifiedUNet(nn.Module):
    def __init__(self, input_channel, retrain=True):
        super().__init__()
        self.retrain = retrain
        self.conv1 = ConvBlock(input_channel=input_channel, output_channel=32, padding=1)
        self.conv2 = ConvBlock(input_channel=32, output_channel=64, padding=1)
        self.conv3 = ConvBlock(input_channel=64, output_channel=128, padding=1)
        self.conv4 = ConvBlock(input_channel=128, output_channel=256, padding=1)
        self.neck = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1
        )
        self.upconv4 = UpConvolution(in_channels=512, out_channels=256)
        self.dconv4 = ConvBlock(input_channel=512, output_channel=256, padding=1)
        self.upconv3 = UpConvolution(in_channels=256, out_channels=128)
        self.dconv3 = ConvBlock(input_channel=256, output_channel=128, padding=1)
        self.upconv2 = UpConvolution(in_channels=128, out_channels=64)
        self.dconv2 = ConvBlock(input_channel=128, output_channel=64, padding=1)
        self.upconv1 = UpConvolution(in_channels=64, out_channels=32)
        self.dconv1 = ConvBlock(input_channel=64, output_channel=32, padding=1)
        self.out = nn.Conv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=1,
            stride=1
        )
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def crop(self, input_tensor, target_tensor):
        _, _, h, w = target_tensor.shape
        cropped = transform.CenterCrop([h, w])(input_tensor)
        return cropped

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.max_pool_2x2(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.max_pool_2x2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.max_pool_2x2(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.max_pool_2x2(conv4)
        neck = self.neck(pool4)
        upconv4 = self.upconv4(neck)
        cropped = self.crop(input_tensor=conv4, target_tensor=upconv4)
        dconv4 = self.dconv4(torch.cat([upconv4, cropped], dim=1))
        upconv3 = self.upconv3(dconv4)
        cropped = self.crop(input_tensor=conv3, target_tensor=upconv3)
        dconv3 = self.dconv3(torch.cat([upconv3, cropped], dim=1))
        upconv2 = self.upconv2(dconv3)
        cropped = self.crop(input_tensor=conv2, target_tensor=upconv2)
        dconv2 = self.dconv2(torch.cat([upconv2, cropped], dim=1))
        upconv1 = self.upconv1(dconv2)
        cropped = self.crop(input_tensor=conv1, target_tensor=upconv1)
        dconv1 = self.dconv1(torch.cat([upconv1, cropped], dim=1))
        out = self.out(dconv1)
        if self.retrain:
            out = F.interpolate(out, list(x.shape)[2:])
        return out

