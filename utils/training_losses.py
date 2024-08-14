import torch


def mse_with_positive_pressure(y_true, y_pred):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * torch.maximum(-y_pred, torch.tensor(0.0, device=y_pred.device))
    return torch.mean(mse + positive_pressure)

