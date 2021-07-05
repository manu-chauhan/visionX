'''Basic utility functions for VisionX repo,
    
    Includes:
    - Calculate the mean and std value of dataset
    - Image Transformations
    - Missclassified code
    - GradCam
    - Custom PyTorch Optimizer
    - Tensorboard usage with PyTorch
    - Progress_bar
    - Format time
'''

import os
import sys
import time
import math
import torch
import torch.nn as nn
import albumentations as A
from matplotlib import pyplot as plt
from albumentations.pytorch.transforms import ToTensor

SEED = 101
BATCH = 128


cuda = None

METRICS = {"train_losses": [], "train_accuracy": [], "test_losses": [], "test_accuracy": []}


def get_device():
    global cuda
    cuda = torch.cuda.is_available()
    return torch.device("cuda" if cuda else "cpu")


def get_dataloader_args(shuffle=True, batch_size=BATCH, num_workers=2):

    # dataloader arguments
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                           pin_memory=True) if cuda else dict(shuffle=shuffle,
                                                              batch_size=num_workers)
    return dataloader_args


def get_mean_sdev(dataset, input_channels: int = 3) -> tuple:
    """
    Calculates the mean and standard deviation of the given dataset.

    :param
        dataset: the dataset that will be passed to PyTorch's DataLoader class for iteration and the calculations
        input_channels: the number of channels in the dataset, default=3 (for RGB images)

    :returns
        tuple of mean and standard deviation (each of typle float)
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             num_workers=2)

    mean = torch.zeros(input_channels)
    sdev = torch.zeros(input_channels)

    for data, target in dataloader:
        for i in range(input_channels):
            mean += data[:, i, :, :].mean()  # since format is B, C, H, W
            sdev += data[:, i, :, :].std()
    mean /= len(dataset)
    sdev /= len(dataset)
    return mean, sdev


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# criterion = nn.CrossEntropyLoss()
def train_eval_model(model, train_loader, optimizer, device, criterion=nn.CrossEntropyLoss, epochs=1,
                     test=False, test_loader=None, scheduler=None, metrics=METRICS):

    train_losses = metrics['train_losses'],
    train_accuracy = metrics['train_accuracy'],
    test_losses = metrics['test_losses'],
    test_accuracy = metrics['test_accuracy']

    model.train()  # set the train mode

    # iterate over for `epochs` epochs and keep storing valuable info

    for epoch in range(epochs):
        correct = processed = train_loss = 0

        print(f"\n epoch num ================================= {epoch+1}")

        pbar = tqdm(train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)  # move data to `device`

            optimizer.zero_grad()  # zero out the gradients to avoid accumulating them over loops

            output = model(data)  # get the model's predictions

            # calculate Negative Log Likelihood loss using ground truth labels and the model's predictions
            loss = criterion(output, target)

            train_loss += loss.item()  # add up the train loss

            loss.backward()  # boom ! The magic function to perform backpropagation and calculate the gradients

            optimizer.step()  # take 1 step for the optimizer and update the weights

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # compare and see how many predictions are coorect and then add up the count
            correct += pred.eq(target.view_as(pred)).sum().item()

            processed += len(data)  # total processed data size

        acc = 100 * correct / processed

        train_losses.append(train_loss)

        train_accuracy.append(acc)

        if scheduler:
            print("\n\n\t\t\tLast LR -->", scheduler.get_last_lr())
            scheduler.step()

        pbar.set_description(desc=f'loss={loss.item()} batch_id={batch_idx}')

        train_loss /= len(train_loader.dataset)
        print('\n\t\t\tTrain metrics: accuracy: {}/{} ({:.4f}%)'.format(correct,
                                                                        len(train_loader.dataset),
                                                                        correct * 100 / len(train_loader.dataset)))

        if test:  # moving to evaluation
            model.eval()  # set the correct mode

            correct = test_loss = 0

            with torch.no_grad():  # to disable gradient calculation with no_grad context

                for data, target in test_loader:

                    data, target = data.to(device), target.to(device)

                    output = model(data)

                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            test_accuracy.append(100. * correct / len(test_loader.dataset))

            print('\n\tTest metrics: average loss: {:.4f}, accuracy: {}/{} ({:.5f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


def plot_graphs(*, name="fig.png", train_acc, train_losses, test_acc, test_losses):
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.savfig(fig)


# transforms
mean, sdev = get_mean_sdev(dataset)
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.14, scale_limit=0.15, rotate_limit=30, p=0.24),
    A.CoarseDropout(max_holes=1, p=0.3, max_height=16,
                    max_width=16, min_holes=1, min_height=16,
                    min_width=16, fill_value=mean),
    # A.MedianBlur(blur_limit=3, p=0.1),
    A.HueSaturationValue(p=0.1),
    #   A.GaussianBlur(blur_limit=3, p=0.12),
    # A.RandomBrightnessContrast(brightness_limit=0.09,contrast_limit=0.1, p=0.15),
    A.Normalize(mean=mean, std=sdev),
    ToTensor()
])

test_transforms = A.Compose([
                            A.Normalize(mean=mean, std=sdev),
                            ToTensor()
                            ])
