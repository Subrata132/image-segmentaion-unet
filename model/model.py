import torch
import torch.nn as nn
import torchvision.transforms as transform


class ConvBlock(nn.Module):
    def __init__(
            self,
            input_channel,
            output_channel,
            kernel=3,
            stride=1,
            padding=0
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class ConvTranspose2d(nn.Module):
    def __init__(
            self,
            input_channel,
            output_channel,
            kernel=2,
            stride=2
    ):
        super().__init__()
        self.conv_transpose_2d = nn.ConvTranspose2d(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=kernel,
            stride=stride
        )

    def forward(self, x):
        x = self.conv_transpose_2d(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channel, retrain=True):
        super().__init__()
        self.conv_en1 = ConvBlock(input_channel=input_channel, output_channel=64)
        self.conv_en2 = ConvBlock(input_channel=64, output_channel=128)
        self.conv_en3 = ConvBlock(input_channel=128, output_channel=256)
        self.conv_en4 = ConvBlock(input_channel=256, output_channel=512)
        self.conv_en5 = ConvBlock(input_channel=512, output_channel=1024)
        self.up_conv_1 = ConvTranspose2d(input_channel=1024, output_channel=512)
        self.conv_de1 = ConvBlock(input_channel=1024, output_channel=512)
        self.up_conv_2 = ConvTranspose2d(input_channel=512, output_channel=256)
        self.conv_de2 = ConvBlock(input_channel=512, output_channel=256)
        self.up_conv_3 = ConvTranspose2d(input_channel=256, output_channel=128)
        self.conv_de3 = ConvBlock(input_channel=256, output_channel=128)
        self.up_conv_4 = ConvTranspose2d(input_channel=128, output_channel=64)
        self.conv_de4 = ConvBlock(input_channel=128, output_channel=64)
        self.conv_1x1 = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1
        )
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def crop(self, input_tensor, target_tensor):
        _, _, h, w = target_tensor.shape
        cropped = transform.CenterCrop([h, w])(input_tensor)
        return cropped

    def forward(self, x):
        conv_en1 = self.conv_en1(x)
        pool1_en1 = self.max_pool_2x2(conv_en1)
        conv_en2 = self.conv_en2(pool1_en1)
        pool1_en2 = self.max_pool_2x2(conv_en2)
        conv_en3 = self.conv_en3(pool1_en2)
        pool1_en3 = self.max_pool_2x2(conv_en3)
        conv_en4 = self.conv_en4(pool1_en3)
        pool1_en4 = self.max_pool_2x2(conv_en4)
        conv_en5 = self.conv_en5(pool1_en4)
        up_conv_1 = self.up_conv_1(conv_en5)
        cropped = self.crop(input_tensor=conv_en4, target_tensor=up_conv_1)
        conv_de1 = self.conv_de1(torch.cat([cropped, up_conv_1], dim=1))
        up_conv_2 = self.up_conv_2(conv_de1)
        cropped = self.crop(input_tensor=conv_en3, target_tensor=up_conv_2)
        conv_de2 = self.conv_de2(torch.cat([cropped, up_conv_2], dim=1))
        up_conv_3 = self.up_conv_3(conv_de2)
        cropped = self.crop(input_tensor=conv_en2, target_tensor=up_conv_3)
        conv_de3 = self.conv_de3(torch.cat([cropped, up_conv_3], dim=1))
        up_conv_4 = self.up_conv_4(conv_de3)
        cropped = self.crop(input_tensor=conv_en1, target_tensor=up_conv_4)
        conv_de4 = self.conv_de4(torch.cat([cropped, up_conv_4], dim=1))
        output = self.conv_1x1(conv_de4)
        return output


