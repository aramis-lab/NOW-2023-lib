import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

from now_2023.datasets import MRIDataset

import warnings
warnings.filterwarnings("ignore")

__all__ = ["CNNModel"]


class CNNModel:
    def __init__(self, learning_rate: float, n_epochs: int, batch_size: int):
        self._learning_rate = learning_rate
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self._network = CustomNetwork()
        if self.use_cuda:
            self._network = self._network.cuda()
        self._criterion = nn.CrossEntropyLoss(reduction='sum')
        self._optimizer = torch.optim.Adam(self._network.parameters(), self._learning_rate)
        self._estimator = None

    def fit(self, img_dir: Path, df: pd.DataFrame, transform=None, batch_size: Optional[int] = None):
        train_dataset = MRIDataset(img_dir, df, transform=transform)
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size or self._batch_size,
            shuffle=True,
            # num_workers=1,
            pin_memory=True,
        )
        self._estimator = train(
            self._network,
            loader,
            self._criterion,
            self._optimizer,
            self._n_epochs,
        )

    def predict(self, img_dir: Path, df: pd.DataFrame, transform=None, batch_size: Optional[int] = None):
        valid_dataset = MRIDataset(img_dir, df, transform=transform)
        loader = DataLoader(
            valid_dataset,
            batch_size=batch_size or self._batch_size,
            shuffle=False,
            #num_workers=1,
            pin_memory=True,
        )
        return test(
            self._estimator,
            loader,
            self._criterion,
        )

    def save(self, output_filename: Path) -> None:
        torch.save(self._estimator, output_filename)



class PadMaxPool3d(nn.Module):
    """A MaxPooling module which deals with odd sizes with padding"""
    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        super(PadMaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool3d(kernel_size, stride, return_indices=return_indices)
        self.pad = nn.ConstantPad3d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [self.stride - f_maps.size(i + 2) % self.stride for i in range(3)]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0

        self.pad.padding = (coords[2], 0, coords[1], 0, coords[0], 0)

        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, indices, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output, indices

        else:
            output = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output


class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            # Size 8@30x40x30
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2),
            # Size 8@15x20x15

            nn.Conv3d(8, 16, 3, padding=1),
            # Size 16@15x20x15
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2),
            # Size 16@8x10x8)

            nn.Conv3d(16, 32, 3, padding=1),
            # Size 32@8x10x8
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            PadMaxPool3d(2, 2),
            # Size 32@4x5x4

        )

        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32 * 4 * 5 * 4, 2)

        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    n_epochs: int,
) -> nn.Module:
    """Train a CNN.

    Parameters
    ----------
    model : nn.Module
        The neural network to train.

    train_loader : DataLoader
        A DataLoader wrapping a MRIDataset.

    criterion : nn.Module
        A method to compute the loss of a mini-batch of images.

    optimizer : torch.optim
        An optimization algorithm.

    n_epochs : int
        The number of epochs performed during training.

    Returns
    -------
    best_model : nn.Module
        The trained neural network.
    """
    from copy import deepcopy

    best_model = deepcopy(model)
    train_best_loss = np.inf

    for epoch in range(n_epochs):
        model.train()
        train_loader.dataset.train()
        for i, data in enumerate(train_loader, 0):
            # Retrieve mini-batch and put data on GPU with .cuda() is possible
            if torch.cuda.is_available():
                images, labels = data['image'].cuda(), data['label'].cuda()
            else:
                images, labels = data['image'], data['label']
            # Forward pass
            outputs = model(images)
            # Loss computation
            loss = criterion(outputs, labels)
            # Back-propagation (gradients computation)
            loss.backward()
            # Parameters update
            optimizer.step()
            # Erase previous gradients
            optimizer.zero_grad()

        _, train_metrics = test(model, train_loader, criterion)

        print(
            f"Epoch {epoch}: loss = {train_metrics['mean_loss']:.4f}, "
            f"balanced accuracy = {train_metrics['balanced_accuracy']:.4f}"
            )

        if train_metrics['mean_loss'] < train_best_loss:
            best_model = deepcopy(model)
            train_best_loss = train_metrics['mean_loss']

    return best_model


def test(model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> tuple:
    """Test a CNN.

    Parameters
    ----------
    model : nn.Module
        The neural network to test.

    data_loader : DataLoader
        A DataLoader wrapping a MRIDataset.

    criterion : nn.Module
        A method to compute the loss of a mini-batch of images.

    Returns
    -------
    results_df : pd.DataFrame
        The label predicted for every subject.

    results_metrics : dict
        A set of metrics.
    """
    model.eval()
    data_loader.dataset.eval()
    columns = ["participant_id", "proba0", "proba1", "true_label", "predicted_label"]
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            if torch.cuda.is_available():
                images, labels = data['image'].cuda(), data['label'].cuda()
            else:
                images, labels = data['image'], data['label']
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(outputs.data, 1)

            for idx, sub in enumerate(data['participant_id']):
                row = [
                    sub,
                    probs[idx, 0].item(),
                    probs[idx, 1].item(),
                    labels[idx].item(),
                    predicted[idx].item(),
                ]
                row_df = pd.DataFrame([row], columns=columns)
                results_df = pd.concat([results_df, row_df])

    results_metrics = compute_metrics(results_df.true_label.values, results_df.predicted_label.values)
    results_df.reset_index(inplace=True, drop=True)
    results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)

    return results_df, results_metrics


def compute_metrics(ground_truth, prediction) -> dict:
    """Computes the accuracy, sensitivity, specificity and balanced accuracy"""
    tp = np.sum((prediction == 1) & (ground_truth == 1))
    tn = np.sum((prediction == 0) & (ground_truth == 0))
    fp = np.sum((prediction == 1) & (ground_truth == 0))
    fn = np.sum((prediction == 0) & (ground_truth == 1))

    metrics_dict = dict()
    metrics_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics_dict['sensitivity'] = 0.0
    if tp + fn != 0:
        metrics_dict['sensitivity'] = tp / (tp + fn)
    metrics_dict['specificity'] = 0.0
    if fp + tn != 0:
        metrics_dict['specificity'] = tn / (fp + tn)

    metrics_dict['balanced_accuracy'] = (metrics_dict['sensitivity'] + metrics_dict['specificity']) / 2

    return metrics_dict
