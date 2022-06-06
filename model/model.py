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
            kernel=3,
            padding=2,
            output_padding=0,
            dilation=1
    ):
        super().__init__()
        self.conv_transpose_2d = nn.ConvTranspose2d(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=kernel,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation
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
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def crop(self, input_tensor, target_tensor):
        _, _, h, w = target_tensor.shape
        cropped = transform.CenterCrop([h, w])(input_tensor)
        return cropped

    def forward(self, x):
        print("Encoder")
        ## block1

        conv_en1 = self.conv_en1(x)
        pool1_en1 = self.max_pool_2x2(conv_en1)
        print(f'{conv_en1.shape, pool1_en1.shape}')

        ## block2
        conv_en2 = self.conv_en2(pool1_en1)
        pool1_en2 = self.max_pool_2x2(conv_en2)
        print(f'{conv_en2.shape, pool1_en2.shape}')

        ## block3
        conv_en3 = self.conv_en3(pool1_en2)
        pool1_en3 = self.max_pool_2x2(conv_en3)
        print(f'{conv_en3.shape, pool1_en3.shape}')

        ## block4
        conv_en4 = self.conv_en4(pool1_en3)
        pool1_en4 = self.max_pool_2x2(conv_en4)
        print(f'{conv_en4.shape, pool1_en4.shape}')

        ## block5
        conv_en5 = self.conv_en5(pool1_en4)
        print(f'{conv_en5.shape}')



