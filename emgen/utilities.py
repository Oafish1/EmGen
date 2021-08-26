from math import floor


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
