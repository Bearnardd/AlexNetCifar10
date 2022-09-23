import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-04, beta=0.75, k=2),
            # Overlapping pooling
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 2
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-04, beta=0.75, k=2),
            # Overlapping pooling
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Overlapping pooling
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def _init_weights(self):
        for layer in self.features:
            nn.init.normal(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)
        # in the paper bias = 1 for 2nd, 4th and 5th conv layers with bias 1
        # as well as all fully connected layers. This initialization is suppose
        # to accelerate the early stages of training by providing ReLUs with
        # positive inputs
        nn.init.constant_(self.features[4].bias, 1)
        nn.init.constant_(self.features[10].bias, 1)
        nn.init.constant_(self.features[12].bias, 1)
        nn.init.constant_(self.classifier[2].bias, 1)
        nn.init.constant_(self.classifier[5].bias, 1)
        nn.init.constant_(self.classifier[7].bias, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
