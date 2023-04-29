import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from data_loader.data_loader import CustomDataLoader
from model.modified_unet import ModifiedUNet
from model.model import UNet
from utils.visuals import show_result


class Trainer:
    def __init__(
            self,
            batch_size=2,
            lr=0.001,
            epochs=30,
            train_image_path='data/train/',
            test_image_path='data/val/',
            show=False
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.train_image_path = train_image_path
        self.test_image_path = test_image_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.show = show

    def train(self, test=False):
        image_transform = transform.Compose(
            [transform.ToTensor()]
        )
        label_transform = transform.Compose(
            [transform.ToTensor()]
        )
        if not test:
            data_loader = CustomDataLoader(
                image_path=self.train_image_path,
                image_transformer=image_transform,
                label_transformer=label_transform
            )
            no_train = int(len(data_loader) * 0.85)
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
            model = UNet(input_channel=3).to(device=self.device)
            loss_func = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            all_train_loss = []
            validation_loss = []
            best_val_loss = 1e10
            best_epoch = 0
            counter = 0
            if not os.path.isdir('saved_weights'):
                os.makedirs('saved_weights')
            for i in range(self.epochs):
                print(f'Epoch: {i + 1}')
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
                all_train_loss.append(train_loss / len(train_data_loader))
                if self.show:
                    show_result(image, output, label)
                print('Validating...........')
                for image, label in tqdm(validation_data_loader):
                    image = image.to(device=self.device)
                    label = label.to(device=self.device)
                    output = model(image)
                    loss = loss_func(output, label)
                    val_loss = val_loss + loss.item()
                current_val_loss = val_loss / len(validation_data_loader)
                validation_loss.append(val_loss / len(validation_data_loader))
                print(f'Epoch: {i + 1} | Train loss: {all_train_loss[-1]} | Validation loss: {validation_loss[-1]}')
                print('')
                model_state = {
                    "state_dict": model.state_dict()
                }
                if current_val_loss <= best_val_loss:
                    torch.save(model_state, f'saved_weights/unet_best_model.pth')
                    best_epoch = i
                    best_val_loss = current_val_loss
                    counter = 0
                else:
                    counter += 1
                if counter > 9:
                    break
            loss_dict = {
                'train_loss': all_train_loss,
                'val_loss': validation_loss
            }

            with open('saved_weights/loss_info.json', 'w') as file:
                json.dump(loss_dict, file)
            file.close()
            print(f'model saved at epoch {best_epoch}')
        else:
            test_data_loader = CustomDataLoader(
                image_path=self.test_image_path,
                image_transformer=image_transform,
                label_transformer=label_transform
            )
            test_data_loader = DataLoader(
                dataset=test_data_loader,
                batch_size=self.batch_size
            )
            model = ModifiedUNet(input_channel=3).to(device=self.device)
            model.load_state_dict(torch.load('saved_weights/best_model.pth')['state_dict'])
            model.eval()
            fig, axes = plt.subplots(self.batch_size, 3, figsize=(18, 7))
            with torch.no_grad():
                for image, label in test_data_loader:
                    image = image.to(device=self.device)
                    label = label.to(device=self.device)
                    output = model(image)
                    images = image.cpu()
                    outputs = output.cpu()
                    labels = label.cpu()
                    k = 0
                    for i, ax in enumerate(axes.ravel()):
                        if i % 3 == 0:
                            ax.imshow(images[k].detach().permute(1, 2, 0).numpy())
                        elif i % 3 == 1:
                            ax.imshow(labels[k].detach().permute(1, 2, 0).numpy())
                        else:
                            ax.imshow(outputs[k].detach().permute(1, 2, 0).numpy())
                            k += 1
                        ax.set_xticks([])
                        ax.set_yticks([])
                    plt.tight_layout()
                    plt.pause(0.5)

