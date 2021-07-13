import torch
import argparse
import torchvision
from pathlib import Path

# to enable benchmark mode
# in order to select best algorithm internally if input size to layers remains same,
# takes a while to start, but helps in faster runtime as per the available hardware and config,
# ref: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
torch.backends.cudnn.benchmark = True

my_parser = argparse.ArgumentParser(
    description='PyTorch based VisionX Training for various Deep Convolutional Vision Models')

required = my_parser.add_argument_group("\n\nRequired Arguments _______________________:\n")
optional = my_parser.add_argument_group("\nOptional Arguments _______________________:\n")

required.add_argument('--data', required=True,
                      type=str,
                      help='Choose the dataset to use (mnist, cifar10)')

required.add_argument('--lr', required=True,
                      default=0.01, type=float, help='learning rate')

optional.add_argument('--batch', type=int, default=128, help='Batch size to use')

optional.add_argument('--model', type=str, default=None, choices=['best', 'custom_resnet', 'resnet18', 'resnet34'],
                      help='model to use')

optional.add_argument('--resume', '-r', action='store_true',
                      help='resume from checkpoint')

optional.add_argument('--device', type=str, help='the device to use to run the model')

optional.add_argument('--lr_finder', type=bool,
                      help='To use lR Finder using One Cycle policy to find best Learning Rate for the model on'
                           ' given dataset, possible values 0 or 1, (type:bool, default: 0),'
                           ' if using this modify `high`, optional argument for maximum epoch to reach `lr`,'
                           ' (default is 0.01)')

optional.add_argument('--high', type=int,
                      help='The highest number of epoch to hit the maximum LR set via `lr` arg. Default: 5 (int)',
                      default=5)

optional.add_argument('--aug', type=bool, default=0,
                      help='Whether to use image augmentation or not (default 0 (int))')

optional.add_argument('--root', type=Path, default='./data', help='The path for dataset to be downloaded')

optional.add_argument('--workers', default=2, type=int, help='Number of workers to use')

args = my_parser.parse_args()

data = args.data.lower()
lr = args.lr
batch = data.batch
resume = data.resume
lr_finder = data.lr_finder
max_one_cycle_epoch = data.high
use_aug = data.aug
workers = data.workers
model = data.model

# `resolve()`: This makes your path absolute and replaces all relative parts with absolute parts,
# and all symbolic links with physical paths.
data_root = args.root.resolve()

# data
if 'cifar10' in data:
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)

    # data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=workers)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
