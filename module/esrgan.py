import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator_2(nn.Module):
    def __init__(self, num_rrdb_blocks: int = 16):
        r""" This is an esrgan model defined by the author himself.

        We use two settings for our generator – one of them contains 8 residual blocks, with a capacity similar
        to that of SRGAN and the other is a deeper model with 16/23 RRDB blocks.

        Args:
            num_rrdb_blocks (int): How many resi dual in residual blocks are combined. (Default: 16).

        Notes:
            Use `num_rrdb_blocks` is 16 for TITAN 2080Ti.
            Use `num_rrdb_blocks` is 23 for Tesla A100.
        """
        super(Generator_2, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 16/23 ResidualInResidualDenseBlock layer.
        trunk = []
        for _ in range(num_rrdb_blocks):
            trunk += [ResidualInResidualDenseBlock(channels=64, growth_channels=32, scale_ratio=0.2)]
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        self.up1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Next layer after upper sampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        trunk = self.trunk(out1)
        out2 = self.conv2(trunk)
        out = torch.add(out1, out2)
        out = F.leaky_relu(self.up1(F.interpolate(out, scale_factor=2, mode="nearest")), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.up2(F.interpolate(out, scale_factor=2, mode="nearest")), negative_slope=0.2, inplace=True)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class ResidualDenseBlock(nn.Module):
    r"""

    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.2)
    """

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        r"""

        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), dim=1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), dim=1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), dim=1))
        conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), dim=1))

        out = torch.add(conv5 * self.scale_ratio, x)

        return out


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined

    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.2)
    """

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)

        out = torch.add(out * self.scale_ratio, x)

        return out