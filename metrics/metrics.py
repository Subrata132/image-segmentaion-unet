import numpy as np
from sklearn.metrics import confusion_matrix


def compute_iou(y_pred, y_true):
    y_pred = list(map(int, 255*y_pred.flatten()))
    y_true = list(map(int, 255*y_true.flatten()))
    labels = np.arange(0, 256, 1)
    current = confusion_matrix(y_true, y_pred, labels=labels)
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / union.astype(np.float32)
    return np.mean(iou)