'''Basic utility functions for VisionX repo,

    Includes:
    - Calculate the mean and std value of dataset
    - Image Transformations
    - Data loader
    - Misclassified code
    - GradCam
    - Custom PyTorch Optimizer
    - Tensorboard usage with PyTorch
    - Progress_bar
    - Format time
'''

import os
import sys
import cv2
import PIL
import time
import copy
import torch
import types
import inspect
import traceback
import torchvision
import numpy as np
from models import *
from tqdm import tqdm
import torch.nn as nn
import albumentations as A
from functools import wraps
from datetime import datetime
from time import perf_counter
import torch.nn.functional as F
from models import custom
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image
from albumentations.pytorch.transforms import ToTensor

import models.resnet

SEED = 101
BATCH = 128

cuda = None

METRICS = {"misclassified": [], "train_losses": [], "train_accuracy": [], "test_losses": [], "test_accuracy": []}

MNIST = "mnist"
CIFAR10 = "cifar10"
CUSTOM_RESNET = "custom_resnet"
RESNET18 = "resnet18"
RESNET34 = "resnet34"


def get_device():
    """
    Check what device is available
    """
    global cuda
    cuda = torch.cuda.is_available()
    return torch.device("cuda" if cuda else "cpu")


def get_dataloader_args(*, shuffle=True, batch_size=BATCH, num_workers=2):
    """
    Create dict for data loader args

    Parameters
    ----------
    shuffle : whether to shuffle the data set
    batch_size : the batch size for each data batch generated from data loader
    num_workers :number of workers to use for parallel data generation

    Returns  dict based on device type available
    -------
    """
    return dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                pin_memory=True) if cuda else dict(shuffle=shuffle,
                                                   batch_size=num_workers)


def get_mean_sdev(*, dataset, input_channels: int = 3) -> tuple:
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


class GradCAM:
    """To calculate GradCAM saliency map.

    Params:
        input: input image with shape of (1, 3, H, W)

        class_idx (int): class index for calculating GradCAM.
                If not specified, the class index that makes the `highest model prediction score` will be used.
    Return:
        mask: saliency map of the same spatial dimension with input

        logit: raw model output

    A simple example:
        # initialize a model, model_dict and gradcam
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index.
        mask, logit = gradcam(normed_img, class_idx=1)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    """

    def __init__(self, model, layer_name):
        self.model = model
        # self.layer_name = layer_name
        self.target_layer = layer_name

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def saliency_map_size(self, *input_size):
        device = next(self.model.parameters()).device
        self.model(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        self.gradients.clear()
        self.activations.clear()
        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


# ------------------------------------VISUALIZE_GRADCAM-------------------------------------------------------------


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


# -------------------------------------------GradCam View (Initialisation)--------------------------------------------


def GradCamView(miscalssified_images, model, classes, layers, Figsize=(23, 30), subplotx1=9, subplotx2=3,
                wantheatmap=False):
    fig = plt.figure(figsize=Figsize)
    for i, k in enumerate(miscalssified_images):
        images1 = [miscalssified_images[i][0].cpu() / 2 + 0.5]
        images2 = [miscalssified_images[i][0].cpu() / 2 + 0.5]
        for j in layers:
            g = GradCAM(model, j)
            mask, _ = g(miscalssified_images[i][0].clone().unsqueeze_(0))
            heatmap, result = visualize_cam(mask, miscalssified_images[i][0].clone().unsqueeze_(0) / 2 + 0.5)
            images1.extend([heatmap])
            images2.extend([result])
        # Ploting the images one by one
        if wantheatmap:
            finalimages = images1 + images2
        else:
            finalimages = images2
        grid_image = make_grid(finalimages, nrow=len(layers) + 1, pad_value=1)
        npimg = grid_image.numpy()
        sub = fig.add_subplot(subplotx1, subplotx2, i + 1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        sub.set_title(
            'P = ' + classes[int(miscalssified_images[i][1])] + " A = " + classes[int(miscalssified_images[i][2])],
            fontweight="bold", fontsize=15)
        sub.axis("off")
        plt.tight_layout()
        fig.subplots_adjust(wspace=0)


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

    L = ['  Step: %s' % format_time(step_time), ' | Tot: %s' % format_time(tot_time)]
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


def train_eval_model(model, train_loader, optimizer, device, criterion=nn.CrossEntropyLoss, epochs=1,
                     test=False, test_loader=None, scheduler=None, metrics=METRICS):
    """
    Train and Eval function
    Parameters
    ----------
    model : the model to use
    train_loader : train data set loader
    optimizer : the optimizer to use for updating the weights
    device : device type( CPU or CUDA)
    criterion : the loss function to use to calculate error value
    epochs : the number of iterations over entire data set
    test : bool value to enable/disable evaluation for the same model
    test_loader : test data loader, must if test is True
    scheduler : a scheduler for dynamic LR
    metrics : a dictionary which contains 4 elements for train & test loss and accuracy, list to store those values
    -------
    """

    train_losses = metrics['train_losses'],
    train_accuracy = metrics['train_accuracy'],
    test_losses = metrics['test_losses'],
    test_accuracy = metrics['test_accuracy']

    model.train()  # set the train mode

    # iterate over for `epochs` epochs and keep storing valuable info

    for epoch in range(epochs):
        correct = processed = train_loss = 0

        print(f"\n epoch num ================================= {epoch + 1}")

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

                    # sum up batch loss
                    test_loss += criterion(output, target).sum().item()

                    # get the index of the max log-probability
                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            test_accuracy.append(100. * correct / len(test_loader.dataset))

            print('\n\tTest metrics: average loss: {:.4f}, accuracy: {}/{} ({:.5f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


def plot_graphs(*, name="fig.png", train_acc, train_losses, test_acc, test_losses):
    """
    Saves the graph for train and test accuracy and loss values with epochs
    -------
    """
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


# # transforms
# mean, sdev = get_mean_sdev(dataset)
# train_transforms = A.Compose([
#     A.HorizontalFlip(p=0.2),
#     A.ShiftScaleRotate(shift_limit=0.14, scale_limit=0.15, rotate_limit=30, p=0.24),
#     A.CoarseDropout(max_holes=1, p=0.3, max_height=16,
#                     max_width=16, min_holes=1, min_height=16,
#                     min_width=16, fill_value=mean),
#     # A.MedianBlur(blur_limit=3, p=0.1),
#     A.HueSaturationValue(p=0.1),
#     #   A.GaussianBlur(blur_limit=3, p=0.12),
#     # A.RandomBrightnessContrast(brightness_limit=0.09,contrast_limit=0.1, p=0.15),
#     A.Normalize(mean=mean, std=sdev),
#     ToTensor()
# ])

# test_transforms = A.Compose([
#                             A.Normalize(mean=mean, std=sdev),
#                             ToTensor()
#                             ])


def get_transforms(*, train=False, normalize=True, mean=None, sdev=None,
                   pre_transforms=None, post_transforms=None):
    """
    Returns a list of transformations based on the params
    Parameters
    ----------
    train : whether the transforms is for train or test data, True for Train
    normalize : if to normalize the values, mean and sdev are must then
    mean : the mean of the dataset, channel wise
    sdev : standard deviation of the dataset, channel wise
    pre_transforms : if any pre-transforms are to be added and used, must be a list or tuple
    post_transforms : if any post-transforms are to be added and used, must be a list or tuple

    Returns  Albumentations' Compose with list of transforms
    -------
    """

    if normalize and not (mean and sdev):
        raise ValueError('mean and sdev both are required for normalize transform')
    if pre_transforms and post_transforms and not (
            isinstance(pre_transforms, (list, tuple)) and isinstance(post_transforms, (list, tuple))):
        raise TypeError("Only list or tuple to be passed for pre or post transforms")
    try:
        transforms_list = [ToTensor()]
        if normalize and not train:
            transforms_list.append(A.Normalize(mean, sdev))
            return A.Compose(transforms_list)
        if pre_transforms:
            transforms_list = list(pre_transforms)
        if normalize:
            transforms_list.append(A.Normalize(mean, sdev))
        if post_transforms:
            transforms_list.extend(list(post_transforms))
        return A.Compose(transforms_list)
    except Exception as e:
        print(traceback.format_exc())
        raise e


def show_image(img, title, figsize=(7, 7), normalize=False, mean=None, sdev=None):
    if normalize and mean and sdev:
        img = img * sdev + mean
    elif normalize:
        img = img / 2 + 0.5
    numpy_img = img.numpy()
    fig = plt.figure(figsize=figsize)
    plt.imshow(np.transpose(numpy_img, (1, 2, 0)), interpolation='none')
    plt.title(title)


def show_training_data(dataset, classes, *, mean=None, sdev=None):
    dataiter = iter(dataset)
    images, labels = next(dataiter)
    # images, labels = images.to('cpu'), labels.to('cpu')
    for i in range(10):
        index = [j for j in range(len(labels)) if labels[j] == i]
        show_image(torchvision.utils.make_grid(images[index[0:5]], nrow=5, padding=2, scale_each=True), classes[i],
                   normalize=True, mean=mean, sdev=sdev)


def show_gradcam(images, model, classes, layers):
    def imshow(img, c=""):
        # img = img / 2 + 0.5
        npimg = img.numpy()
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='none')
        plt.title(c)

    pil_image = []
    for i, img in enumerate(images):
        pil_image.append(PIL.Image.open(img))

    normed_torch_img = []
    torch_img_list = []

    for i in pil_image:
        torch_img = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()])(i).to(get_device())
        torch_img_list.append(torch_img)
        normed_torch_img.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(torch_img)[None])

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for i, k in enumerate(normed_torch_img):
        images1 = [torch_img_list[i].cpu()]
        images2 = [torch_img_list[i].cpu()]
        b = copy.deepcopy(model.to(get_device()))
        output = model(normed_torch_img[i])
        _, predicted = torch.max(output.data, 1)

        '''first 3 layers and their sub-layers with size MORE THAN 7x7'''
        layers = [b.layer1[0], b.layer1[1], b.layer2[0], b.layer2[1], b.layer3[0], b.layer3[1]]
        for j in layers:
            g = GradCAM(b, j)
            mask, _ = g(normed_torch_img[i])
            heatmap, result = visualize_cam(mask, torch_img_list[i])
            images1.extend([heatmap])
            images2.extend([result])
        grid_image = make_grid(images1 + images2, nrow=7)
        imshow(grid_image, c=classes[int(predicted)])


def logger(fn: "Function"):
    """
    A logger decorator to `log` any function decorated with it, providing essential information such as:
    time of run, total execution time, description, signature, arguments.
    """
    if not isinstance(fn, types.FunctionType):
        raise TypeError("😑 The passed `fn` is not a function, kindly pass a VALID function")

    @wraps(fn)
    def inner(*args, **kwargs):
        all_args = [repr(x) for x in args]
        all_kwargs = [f"{k}={v!r}" for k, v in kwargs.items()]
        combined_args = ",".join(all_args + all_kwargs)
        print(f"Function with function name: {fn.__name__}({combined_args})\n called at"
              f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        start = perf_counter()
        result = fn(*args, **kwargs)
        end = perf_counter()
        print(f"Function description:\n{fn.__doc__}")
        print(f"Function annotations with signature : {inspect.signature(fn)}")
        print(
            f"Execution time to run {fn.__name__} is {(end - start)} and the result returned : {result}\n\n------------------------------")
        return result

    return inner


def resolve_model(data: str, model_name: str):
    """
    To resolve the model name to actual PyTorch model

    Parameters
    ----------
    data : dataset to use, use in order to select custom or best model for specific dataset if available
    model_name : the string for model to use

    Returns
    -------
    The network to use, moved to available device, after check
    """

    data = data.lower().strip()
    model_name = model_name.lower().strip()
    device = get_device()
    if CIFAR10 in data:
        if CUSTOM_RESNET in model_name:
            return custom.CUSTOM_RESNET_CIFAR10.to(device)
        if RESNET18 in model_name:
            return models.resnet.ResNet18().to(device)
        if RESNET34 in model_name:
            return models.resnet.ResNet34().to(device)


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
