from torch import nn

from ..backend import EmGenModel


class Downscale(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()

        self.conv = nn.Conv2d(inputs, outputs, 1, stride=2)
        self.bn = nn.BatchNorm2d(outputs)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        return out


class BasicBlock(nn.Module):
    # Converted from
    # https://www.pluralsight.com/guides/introduction-to-resnet
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L300
    def __init__(self, inputs, outputs):
        super().__init__()

        should_downscale = inputs != outputs
        self.downscale = (
            Downscale(inputs, outputs)
            if should_downscale else None
        )
        self.conv1 = nn.Conv2d(inputs,
                               outputs,
                               3,
                               stride=2 if should_downscale else 1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outputs, outputs, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outputs)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downscale:
            identity = self.downscale(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(EmGenModel):
    def __init__(self):
        super().__init__()

        def main_block(inputs, outputs):
            return nn.Sequential(
                BasicBlock(inputs, outputs),
                BasicBlock(outputs, outputs),
            )

        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.blocks = nn.ModuleList([
            main_block(64, 64),
            main_block(64, 128),
            main_block(128, 256),
            main_block(256, 512),
        ])
        self.postprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = x.to(self.device)
        out = x.permute(0, 3, 1, 2)

        out = self.preprocess(out)
        for block in self.blocks:
            out = block(out)
        out = self.postprocess(out)

        return out
