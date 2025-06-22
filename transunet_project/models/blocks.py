import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Standard convolutional block: Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Basic Residual Block for ResNet"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class UpsampleBlock(nn.Module):
    """Upsampling block: Upsample (Bilinear) -> ConvBlock"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1) # Input channels might need adjustment if concatenating

    def forward(self, x, skip_features=None):
        x = self.upsample(x)
        if skip_features is not None:
            # Assuming x and skip_features have the same H, W
            # Adjust channels for conv_block if concatenating
            # For now, let's assume conv_block's in_channels is correctly set for x after upsampling
            # Concatenation will be handled in the main TransUNet decoder
            pass
        return self.conv_block(x)

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) for feature map filtering in skip connections.
    As described in "Attention U-Net: Learning Where to Look for the Pancreas".
    """
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: Number of channels in the gating signal (from the coarser scale, upsampled decoder path).
        F_l: Number of channels in the input signal from the skip connection (encoder path).
        F_int: Number of channels in the intermediate (bottleneck) layer.
               A common choice is F_l // 2 or F_g // 2.
        """
        super(AttentionGate, self).__init__()

        # Ensure F_int is at least 1, especially if F_l or F_g are small.
        # The AttentionGate constructor in DecoderBlock will set F_int = max(1, skip_channels // 2)
        _F_int = max(1, F_int)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, _F_int, kernel_size=1, stride=1, padding=0, bias=False), # Bias False since BN follows
            nn.BatchNorm2d(_F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, _F_int, kernel_size=1, stride=1, padding=0, bias=False), # Bias False since BN follows
            nn.BatchNorm2d(_F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(_F_int, 1, kernel_size=1, stride=1, padding=0, bias=False), # Bias False since BN follows
            nn.BatchNorm2d(1), # Output is 1 channel, then sigmoid
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: Gating signal from the coarser scale (e.g., upsampled features from previous decoder layer).
           Expected shape: (B, F_g, H, W)
        x: Input features from the skip connection (e.g., features from corresponding encoder layer).
           Expected shape: (B, F_l, H, W)
        It's assumed that g and x have the same spatial dimensions (H, W) before being passed to this gate.
        If not, spatial alignment (e.g., F.interpolate) should be done by the caller.
        """
        g1 = self.W_g(g)  # Transform gating signal -> (B, F_int, H, W)
        x1 = self.W_x(x)  # Transform input skip signal -> (B, F_int, H, W)

        # Add transformed signals and apply ReLU
        # g1 and x1 must have same H, W for addition.
        # If g was upsampled and x is from encoder, they might need alignment.
        # This check is a safeguard, but alignment is better handled by the caller.
        if g1.shape[2:] != x1.shape[2:]:
            # Typically, g (from upsampling) is aligned to x (from encoder skip).
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi_intermediate = self.relu(g1 + x1) # (B, F_int, H, W)

        # Compute attention coefficients (alpha)
        alpha = self.psi(psi_intermediate)    # (B, 1, H, W)

        # Multiply input signal x by attention coefficients
        return x * alpha                     # (B, F_l, H, W)
