import torch.nn as nn


class CustomResNetCifar10(nn.Module):

    @classmethod
    def get_conv_block(cls, in_channels=3, out_channels=64, max_pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if max_pool:
            layers.append(nn.MaxPool2d(2))

        layers.extend([nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)])

        return nn.Sequential(*layers)

    def __init__(self, in_planes, num_classes):
        super().__init__()

        self.conv1 = self.get_conv_block(in_channels=in_planes, out_channels=64)
        self.conv2 = self.get_conv_block(in_channels=64, out_channels=128, max_pool=True)
        self.res1 = nn.Sequential(self.get_conv_block(in_channels=128, out_channels=128),
                                  self.get_conv_block(in_channels=128, out_channels=128))

        self.conv3 = self.get_conv_block(128, 256, max_pool=True)
        self.conv4 = self.get_conv_block(256, 512, max_pool=True)
        self.res2 = nn.Sequential(self.get_conv_block(512, 512),
                                  self.get_conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.1),
                                        nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.res1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out + self.res2(out)
        out = self.classifier(out)
        return out


CUSTOM_RESNET_CIFAR10 = CustomResNetCifar10(3, 10)
