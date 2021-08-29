from math import floor

from torch import nn


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


def conv_output_shape(dim, kernel_size=1, stride=1, padding=0, dilation=1):
    # Adapted from
    # https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(padding) is not tuple:
        padding = (padding, padding)
    if type(dilation) is not tuple:
        dilation = (dilation, dilation)
    h = floor(
        (dim[1] + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1)
        / stride[0]
    ) + 1
    w = floor(
        (dim[2] + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1)
        / stride[1]
    ) + 1
    return dim[0], h, w


def pool_output_shape(dim, kernel_size=1, stride=None, padding=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif type(stride) is not tuple:
        stride = (stride, stride)
    if type(padding) is not tuple:
        padding = (padding, padding)
    if type(dilation) is not tuple:
        dilation = (dilation, dilation)
    h = floor(
        (dim[1] + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1)
        / stride[0]
    ) + 1
    w = floor(
        (dim[2] + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1)
        / stride[1]
    ) + 1
    return dim[0], h, w
