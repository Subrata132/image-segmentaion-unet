import numpy as np
from sklearn.metrics import confusion_matrix


def compute_iou(y_pred, y_true):
    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    current = confusion_matrix(y_true, y_pred, labels=labels)
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / union.astype(np.float32)
    return np.mean(iou)