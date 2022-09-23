import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torch.optim import Optimizer
from torchvision import transforms, datasets
from tqdm import tqdm

from typing import List, Optional, Union


def _compute_val_acc(net: nn.Module, loader: DataLoader, device: torch.device):
    with torch.no_grad():
        correct_preds, num_examples = 0, 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net(inputs)

            _, pred_labels = torch.max(outputs, 1)

            num_examples += targets.size(0)
            correct_preds += (pred_labels == targets).sum()
    return correct_preds.float() / num_examples * 100


def get_train_val_dataloaders(
    batch_size: int,
    fraction_of_data: float,
    val_size: float = 0.1,
    train_transforms: Optional[
        Union[transforms.Compose, transforms.ToTensor]
    ] = transforms.ToTensor(),
):

    train_dataset = datasets.CIFAR10(
        root="data", train=True, transform=train_transforms, download=True
    )

    val_dataset = datasets.CIFAR10(
        root="data", train=True, transform=train_transforms, download=False
    )

    data_size = len(train_dataset)
    indices = torch.randperm(data_size)
    limit_idx = data_size - int(data_size * val_size)
    train_ids = indices[:limit_idx]
    val_ids = indices[limit_idx:]
    if fraction_of_data < 1:
        import numpy as np
        fraction_size = int(data_size * fraction_of_data)
        train_ids = np.random.choice(train_ids, fraction_size)
        val_ids = np.random.choice(val_ids, fraction_size)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        # data will be loaded in the main process
        num_workers=0,
        sampler=train_ids,
    )



    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        # data will be loaded in the main process
        num_workers=0,
        sampler=val_ids,
    )

    return train_loader, val_loader


def train_model(
    net: nn.Module,
    num_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion,
    lr_scheduler,
    device,
    verbose,
    log_freq
):
    minibatch_loss_list, train_acc_list, val_acc_list = [], [], []
    if verbose:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
        print("Logger created!")

    for epoch in range(num_epochs):
        # per epoch
        correct_preds, num_examples = 0, 0
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # predict
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()

            # backpropagate loss
            loss.backward()

            # update parameters
            optimizer.step()

            _, pred_labels = torch.max(outputs, 1)

            num_examples += targets.size(0)
            correct_preds += (pred_labels == targets).sum()

            # logging
            minibatch_loss_list.append(loss.item())
            if verbose and not batch_idx % log_freq:
                print(
                    f"Epoch: {epoch}/{num_epochs}\nBatch: {batch_idx}/{len(train_loader)}\nLoss:{loss:.4f}"
                )

            iteration = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/Train", loss.item(), iteration)

        net.eval()
        with torch.no_grad():
            # train accuracy
            train_acc = correct_preds.float() / num_examples * 100

            # validation accuracy
            val_acc = _compute_val_acc(net, val_loader, device)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)

            train_acc_list.append(train_acc.item())
            val_acc_list.append(val_acc.item())
            if verbose:
                print(
                    f"Epoch: {epoch}/{num_epochs}\nTrain Accuracy: {train_acc:.4f}\nValidation Accuracy:{val_acc:.4f}"
                )

        if lr_scheduler:
            lr_scheduler.step(val_acc_list[-1])

    return minibatch_loss_list, train_acc_list, val_acc_list
