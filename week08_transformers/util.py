import math
import torch

import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from IPython.display import clear_output


# useful utility class for computing averages
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, optimizer, loader, scheduler=None, device='cpu'):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # get accuracy
        acc = torch.eq(outputs.view(-1) > 0.0, targets).float().mean()
        # update stats
        loss_m.update(loss.item(), inputs.shape[0])
        acc_m.update(acc.item(), inputs.shape[0])
        # we use step-wise scheduler
        if scheduler is not None:
            scheduler.step()
    return loss_m.avg, acc_m.avg


@torch.no_grad()
def val_epoch(model, loader, device='cpu'):
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        # get accuracy
        acc = torch.eq(outputs.view(-1) > 0.0, targets).float().mean()
        # update stats
        loss_m.update(loss.item(), inputs.shape[0])
        acc_m.update(acc.item(), inputs.shape[0])
    return loss_m.avg, acc_m.avg


def plot_history(train_losses, train_accs, val_losses, val_accs, figsize=(12, 6)):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax[0].plot(train_losses, label='train')
    ax[0].plot(val_losses, label='val')
    ax[0].set_xlabel('Epoch', fontsize=16)
    ax[0].set_ylabel('Loss', fontsize=16)
    ax[0].legend()

    ax[1].plot(train_accs, label='train')
    ax[1].plot(val_accs, label='val')
    ax[1].set_xlabel('Epoch', fontsize=16)
    ax[1].set_ylabel('Accuracy', fontsize=16)
    ax[1].legend()

    fig.tight_layout()
    plt.show()


def train(
    model,
    num_epochs,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device='cpu'
):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for i in tqdm(range(num_epochs)):
        # run train epoch
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, scheduler, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        # run val epoch
        val_loss, val_acc = val_epoch(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        clear_output()
        plot_history(train_losses, train_accs, val_losses, val_accs)


# cosine annealing LR schedule with Warmup
class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_steps: int, max_steps: int):
        self.warmup = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_steps))
        lr_factor *= min(epoch / self.warmup, 1.0)
        return lr_factor


def hardcode_parameters(module: nn.Module):
  for i, layer in enumerate(module.modules()):
    if isinstance(layer, nn.Linear):
        dim_out, dim_in = layer.weight.shape
        layer.weight.data = torch.cos(i * torch.arange(dim_out))[:, None] \
          * torch.cos(i * torch.arange(dim_in))[None, :]
        if layer.bias is not None:
            layer.bias.data.fill_(0)
