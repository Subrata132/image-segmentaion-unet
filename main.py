import torch
from torch.utils.data import DataLoader
from data_loader.data_loader import CustomDataLoader
from model.model import UNet
from utils.visuals import show_result


def main():
    # data_loader = CustomDataLoader(
    #     image_path='data/train/'
    # )
    # batch_size = 4
    # data_loader = DataLoader(data_loader, batch_size=batch_size)
    # image, label = next(iter(data_loader))
    # print(f'Image type: {type(image)} Image shape: {image.shape}')
    # print(f'Label type: {type(label)} Label shape: {label.shape}')
    # # show_result(image, label)
    image = torch.rand((4, 1, 572, 572))
    unet = UNet(input_channel=1)
    unet(image)


if __name__ == '__main__':
    main()