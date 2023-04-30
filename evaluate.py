import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transform
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader.data_loader import CustomDataLoader
from model.modified_unet import ModifiedUNet
from metrics.metrics import compute_iou
from torchmetrics import JaccardIndex, Dice
from torchmetrics.classification import MulticlassJaccardIndex
from data_loader.pixel_fixer import PixelFixer


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
    fixer = PixelFixer()
    model = ModifiedUNet(input_channel=3).to(device=device)
    model.load_state_dict(torch.load('saved_weights/best_model.pth')['state_dict'])
    model.eval()
    mses = []
    dices = []
    # iou = JaccardIndex(task="multiclass", num_classes=256)
    # iou = MulticlassJaccardIndex(num_classes=256, average="weighted")
    # dice = Dice(average='micro')
    ious = []
    loss_func = nn.MSELoss()
    with torch.no_grad():
        for image, label in tqdm(test_data_loader):
            image = image.to(device=device)
            label = label.to(device=device)
            output = model(image)
            images = image.cpu()
            outputs = output.cpu()
            labels = label.cpu()
            for i in range(outputs.shape[0]):
                pred = torch.clamp((torch.mul(outputs[0], 255)).int(), min=0, max=255)
                actual = (torch.mul(labels[0], 255)).int()
                _, y_true = fixer.fix(actual)
                _, y_pred = fixer.fix(pred)
                ious.append(compute_iou(y_true=y_true, y_pred=y_pred))
            # ious.append(iou(pred, actual))
            # dices.append(dice(pred, actual))
            mses.append(loss_func(output, label).item())

    print(np.mean(mses), np.std(mses))
    print(np.mean(ious), np.std(ious))


if __name__ == '__main__':
    evaluate()