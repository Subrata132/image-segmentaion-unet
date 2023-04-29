import torch
import numpy as np
import torchvision.transforms as transform
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader.data_loader import CustomDataLoader
from model.modified_unet import ModifiedUNet
from metrics.metrics import compute_iou
from torchmetrics import JaccardIndex, Dice


def evaluate():
    test_image_path = 'data/val/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    image_transform = transform.Compose(
        [transform.ToTensor()]
    )
    label_transform = transform.Compose(
        [transform.ToTensor()]
    )
    test_data_loader = CustomDataLoader(
        image_path=test_image_path,
        image_transformer=image_transform,
        label_transformer=label_transform
    )
    test_data_loader = DataLoader(
        dataset=test_data_loader,
        batch_size=batch_size
    )
    model = ModifiedUNet(input_channel=3).to(device=device)
    model.load_state_dict(torch.load('saved_weights/saved_model_20.pth')['state_dict'])
    model.eval()
    ious = []
    dices = []
    iou = JaccardIndex(task="multiclass", num_classes=256)
    dice = Dice(average='micro')
    with torch.no_grad():
        for image, label in tqdm(test_data_loader):
            image = image.to(device=device)
            label = label.to(device=device)
            output = model(image)
            images = image.cpu()
            outputs = output.cpu()
            labels = label.cpu()
            # for i in range(outputs.shape[0]):
            pred = torch.clamp((torch.mul(outputs, 255)).int(), min=0, max=255)
            actual = (torch.mul(labels, 255)).int()
            ious.append(iou(pred, actual))
            # dices.append(dice(pred, actual))

    print(np.mean(ious), np.mean(dices))


if __name__ == '__main__':
    evaluate()