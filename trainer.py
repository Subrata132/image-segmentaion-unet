from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transform
from data_loader.data_loader import CustomDataLoader
from model.modified_unet import ModifiedUNet
from utils.visuals import show_result


class Trainer:
    def __init__(
            self,
            batch_size=2,
            lr=0.01,
            epochs=30,
            image_path='data/train/'
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.image_path = image_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        image_transform = transform.Compose(
            [transform.ToTensor()]
        )
        label_transform = transform.Compose(
            [transform.ToTensor()]
        )
        data_loader = CustomDataLoader(
            image_path=self.image_path,
            image_transformer=image_transform,
            label_transformer=label_transform
        )
        no_train = int(len(data_loader)*0.85)
        no_valid = len(data_loader) - no_train
        train_data_loader, validation_data_loader = torch.utils.data.random_split(
            dataset=data_loader,
            lengths=[no_train, no_valid]
        )
        train_data_loader = DataLoader(
            dataset=train_data_loader,
            batch_size=self.batch_size
        )
        validation_data_loader = DataLoader(
            dataset=validation_data_loader,
            batch_size=self.batch_size
        )
        model = ModifiedUNet(input_channel=3).to(device=self.device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        all_train_loss = []
        validation_loss = []
        for i in range(self.epochs):
            print(f'Epoch: {i+1}')
            train_loss = 0
            val_loss = 0
            print('Training.............')
            for image, label in tqdm(train_data_loader):
                optimizer.zero_grad()
                image = image.to(device=self.device)
                label = label.to(device=self.device)
                output = model(image)
                loss = loss_func(output, label)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss.item()
            all_train_loss.append(train_loss/len(train_data_loader))
            show_result(image, output, label)
            print('Validating...........')
            for image, label in tqdm(validation_data_loader):
                image = image.to(device=self.device)
                label = image.to(device=self.device)
                output = model(image)
                loss = loss_func(output, label)
                val_loss = val_loss + loss.item()
            validation_loss.append(val_loss/len(validation_data_loader))
            print(f'Epoch: {i+1} | Train loss: {all_train_loss[-1]} | Validation loss: {validation_loss[-1]}')
            print('')

