import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import ReverseLayerF
of_shape = 2352


class Extractor(nn.Module):
    def __init__(self, size='small'):
        super(Extractor, self).__init__()
        assert size in ['small', 'middle', 'large', 'mixed'], \
            "arch could be only in ['small', 'middle', 'large', 'mixed']"

        self.size = size
        print(f'{self.size} net'.upper())

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.extractor_middle = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.extractor_big = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=96, out_channels=144, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=144, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        if self.size == 'small':
            x = self.extractor(x)
            x = x.view(-1, 3 * 28 * 28)
        elif self.size == 'middle':
            x = self.extractor_middle(x)
            x = x.view(-1, 128 * 6 * 6)
        else:
            x = self.extractor_big(x)
            x = x.view(-1, 256 * 3 * 3)  # 2304
        return x


class Classifier(nn.Module):
    def __init__(self, size='small'):
        super(Classifier, self).__init__()
        self.size = size
        self.classifier = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10)
        )

        self.classifier_middle = nn.Sequential(
            nn.Linear(in_features=128 * 6 * 6, out_features=3072),
            nn.ReLU(),
            nn.Linear(in_features=3072, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=10)
        )

        self.classifier_big = nn.Sequential(
            nn.Linear(in_features=256 * 3 * 3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10)
        )
        self.classifier_mixed = nn.Sequential(
            nn.Linear(in_features=256 * 3 * 3, out_features=3072),
            nn.ReLU(),
            nn.Linear(in_features=3072, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=10)
        )

    def forward(self, x):
        if self.size == 'small':
            x = self.classifier(x)
        elif self.size == 'middle':
            x = self.classifier_middle(x)
        elif self.size == 'large':
            x = self.classifier_big(x)
        else:
            x = self.classifier_mixed(x)
        return F.softmax(x)


class Discriminator(nn.Module):
    def __init__(self, size='small'):
        super(Discriminator, self).__init__()
        self.size = size
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )
        self.discriminator_middle = nn.Sequential(
            nn.Linear(in_features=128 * 6 * 6, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2)
        )
        self.discriminator_big = nn.Sequential(
            nn.Linear(in_features=256 * 3 * 3, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        if self.size == 'small':
            x = self.discriminator(reversed_input)
        elif self.size == 'middle':
            x = self.discriminator_middle(reversed_input)
        else:
            x = self.discriminator_big(reversed_input)
        return F.softmax(x)


def weights_loader(arch, block, path):
    if block in path:
        if path[block] is not None and path[block] != '':
            if os.path.exists(path[block]):
                weights = torch.load(path[block])
                arch.load_state_dict(weights)
                print(f'{block} Weights loaded')
            else:
                print('weights not found..')
    return arch
