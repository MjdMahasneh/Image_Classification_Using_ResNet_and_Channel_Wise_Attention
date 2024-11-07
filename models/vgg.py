import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import time
import numpy as np


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm):
        super().__init__()

        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class VGG11(nn.Module):
    def __init__(self, output_dim, block, pool, batch_norm):
        super().__init__()

        self.features = nn.Sequential(
            block(3, 64, batch_norm),  # in_channels, out_channels
            pool(2, 2),  # kernel_size, stride
            block(64, 128, batch_norm),
            pool(2, 2),
            block(128, 256, batch_norm),
            block(256, 256, batch_norm),
            pool(2, 2),
            block(256, 512, batch_norm),
            block(512, 512, batch_norm),
            pool(2, 2),
            block(512, 512, batch_norm),
            block(512, 512, batch_norm),
            pool(2, 2),
        )

        self.classifier = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, output_dim, block, pool, batch_norm):
        super().__init__()

        self.features = nn.Sequential(
            block(3, 64, batch_norm),
            block(64, 64, batch_norm),
            pool(2, 2),
            block(64, 128, batch_norm),
            block(128, 128, batch_norm),
            pool(2, 2),
            block(128, 256, batch_norm),
            block(256, 256, batch_norm),
            block(256, 256, batch_norm),
            pool(2, 2),
            block(256, 512, batch_norm),
            block(512, 512, batch_norm),
            block(512, 512, batch_norm),
            pool(2, 2),
            block(512, 512, batch_norm),
            block(512, 512, batch_norm),
            block(512, 512, batch_norm),
            pool(2, 2),
        )

        self.classifier = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class VGG19(nn.Module):
    def __init__(self, output_dim, block, pool, batch_norm):
        super().__init__()

        self.features = nn.Sequential(
            block(3, 64, batch_norm),
            block(64, 64, batch_norm),
            pool(2, 2),
            block(64, 128, batch_norm),
            block(128, 128, batch_norm),
            pool(2, 2),
            block(128, 256, batch_norm),
            block(256, 256, batch_norm),
            block(256, 256, batch_norm),
            block(256, 256, batch_norm),
            pool(2, 2),
            block(256, 512, batch_norm),
            block(512, 512, batch_norm),
            block(512, 512, batch_norm),
            block(512, 512, batch_norm),
            pool(2, 2),
            block(512, 512, batch_norm),
            block(512, 512, batch_norm),
            block(512, 512, batch_norm),
            block(512, 512, batch_norm),
            pool(2, 2),
        )

        self.classifier = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':

    OUTPUT_DIM = 10
    BATCH_NORM = True

    vgg11_model = VGG11(OUTPUT_DIM, VGGBlock, nn.MaxPool2d, BATCH_NORM)
    vgg16_model = VGG16(OUTPUT_DIM, VGGBlock, nn.MaxPool2d, BATCH_NORM)
    vgg19_model = VGG19(OUTPUT_DIM, VGGBlock, nn.MaxPool2d, BATCH_NORM)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'VGG11 has {count_parameters(vgg11_model):,} trainable parameters')
    print(f'VGG16 has {count_parameters(vgg16_model):,} trainable parameters')
    print(f'VGG19 has {count_parameters(vgg19_model):,} trainable parameters')


    OUTPUT_DIM = 10
    BATCH_NORM = True

    vgg11_model = VGG11(OUTPUT_DIM, VGGBlock, nn.MaxPool2d, BATCH_NORM)
    vgg16_model = VGG16(OUTPUT_DIM, VGGBlock, nn.MaxPool2d, BATCH_NORM)
    vgg19_model = VGG19(OUTPUT_DIM, VGGBlock, nn.MaxPool2d, BATCH_NORM)

    model = VGG11(OUTPUT_DIM, VGGBlock, nn.MaxPool2d, BATCH_NORM)
    print(model)