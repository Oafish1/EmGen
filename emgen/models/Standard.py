from math import floor, prod

from torch import nn

from ..backend import EmGenModel


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


class Standard(EmGenModel):
    def __init__(self,
                 input_size=128,
                 conv_depth=64,
                 conv_kernel=5,
                 pool_scale=2,
                 num_conv_layers=4,
                 embed_size=50):
        super().__init__()

        self.input_size = input_size
        self.conv_depth = conv_depth
        self.conv_kernel = conv_kernel
        self.pool_scale = pool_scale
        self.num_conv_layers = num_conv_layers
        self.embed_size = embed_size

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(self.pool_scale)
        self.flatten = nn.Flatten()

        self.conv_layers = [nn.Conv2d(3 if i == 0 else self.conv_depth,
                                      self.conv_depth,
                                      self.conv_kernel)
                            for i in range(num_conv_layers)]
        self.batchnorm_layers = [nn.BatchNorm2d(self.conv_depth)
                                 for i in range(num_conv_layers)]
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.batchnorm_layers = nn.ModuleList(self.batchnorm_layers)

        self.dropout = nn.Dropout(.8)

        dim = (3, input_size, input_size)
        for i in range(num_conv_layers):
            dim = self.conv_depth, *dim[1:]
            dim = conv_output_shape(dim, self.conv_kernel)
            dim = pool_output_shape(dim, self.pool_scale)
        calc_dim = prod(dim)
        self.linear = nn.Linear(calc_dim, self.embed_size)

    def forward(self, x):
        x = x.to(self.device)
        out = x.permute(0, 3, 1, 2)
        for conv, batchnorm in zip(self.conv_layers, self.batchnorm_layers):
            out = conv(out)
            out = self.relu(out)
            out = self.maxpool(out)
            out = batchnorm(out)
        out = self.flatten(out)
        out = self.dropout(out)
        out = self.linear(out)
        return out
