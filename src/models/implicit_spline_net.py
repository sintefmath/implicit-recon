import torch.nn as nn
import torch
import torchvision


class ConvBlock(nn.Module):
    """
    Torch neural network module of a convolutional block used in the UNetImplicit architecture.

    :param int in_ch : The number of input channels of the block.
    :param int out_ch: The number of output channels of the block.
    """
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UNetImplicit(nn.Module):
    """
    Torch neural network module of the UNetImplicit architecture.

    :param namespace config: Namespace containing scalings (e.g. 4), code_size (e.g. 8), filters (e.g. 64),
                             and num_input_slices (e.g. 1).
    """
    def __init__(self, config):
        super(UNetImplicit, self).__init__()
        self.down_scalings = config.scalings
        self.up_scalings = config.scalings
        self.code_size = config.code_size
        self.filters = config.filters
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ad_avg_pools = [nn.AdaptiveAvgPool2d((self.code_size * 2 ** i, self.code_size * 2 ** i)) for i in
                             range(self.up_scalings + 1)]

        self.conv_in = ConvBlock(config.num_input_slices, self.filters)
        self.left_convs = nn.ModuleList([ConvBlock((2 ** i) * self.filters, (2 ** (i + 1)) * self.filters)
                                         for i in range(self.down_scalings)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(int(self.filters * (2 ** (self.down_scalings - i))),
                                                     int(self.filters * (2 ** (self.down_scalings - i - 1))),
                                                     kernel_size=(2, 2), stride=(2, 2))
                                  for i in range(self.up_scalings)])
        self.right_convs = nn.ModuleList([ConvBlock(int(self.filters * (2 ** (self.down_scalings - i))),
                                                    int(self.filters * (2 ** (self.down_scalings - i - 1))))
                                          for i in range(self.up_scalings)])

        self.conv_out = nn.Conv2d(self.filters, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x_in):
        x = [self.conv_in(x_in)]

        skips = []
        for i in range(self.down_scalings):
            skips.append(x[-1])
            x.append(self.max_pool(x[-1]))
            x.append(self.left_convs[i](x[-1]))

        x.append(self.ad_avg_pools[0](x[-1]))

        for i in range(self.up_scalings):
            x.append(self.ups[i](x[-1]))
            x.append(torch.cat((self.ad_avg_pools[i + 1](skips[self.down_scalings - i - 1]), x[-1]), dim=1))
            x.append(self.right_convs[i](x[-1]))

        x.append(self.conv_out(x[-1]))

        return x[-1]


class VGGTrunc(nn.Sequential):
    """
    Torch neural network module (sequential) of the VGGTrunc architecture.

    :param namespace config: Namespace with a variable (string) network indicating the variant of VGGTrunc.
    """
    def __init__(self, config):
        VGG16 = torchvision.models.vgg16_bn(pretrained=True, progress=True)
        VGG16_modules = list(VGG16.children())[:-2][0]
        convs_modules = list(VGG16_modules.children())

        # Adapt first layer according to the number of input channels
        layer1 = nn.Conv2d(config.num_input_slices, config.filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if config.network == 'VGGTrunc1':
            super(VGGTrunc, self).__init__(*([layer1] + list(convs_modules)[1:-21]) +
                           [nn.Conv2d(256, 16, kernel_size=(1, 1)), nn.BatchNorm2d(16), nn.Tanh()] +
                           [nn.Conv2d(16, 1, kernel_size=(1, 1))])
        elif config.network == 'VGGTrunc2':
            super(VGGTrunc, self).__init__(*([layer1] + list(convs_modules)[1:-11]) +
                           [nn.Conv2d(512, 32, kernel_size=(1, 1)), nn.BatchNorm2d(32), nn.Tanh()] +
                           [nn.Conv2d(32, 1, kernel_size=(1, 1))])
