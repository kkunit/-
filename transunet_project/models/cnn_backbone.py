import torch
import torch.nn as nn
from .blocks import ResidualBlock, ConvBlock
import logging

logger = logging.getLogger(__name__)

class ResNetBackbone(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[3, 4, 6, 3],
                 zero_init_residual=False, in_channels=3,
                 pretrained_weights_path=None, freeze_stages=-1):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d # Using BatchNorm2d as specified in many ResNet variants

        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # This backbone is for feature extraction, so avgpool and fc are removed.
        # We will extract features from intermediate layers.

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock): # Assuming Bottleneck or BasicBlock
                    nn.init.constant_(m.bn2.weight, 0)

        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)

        if freeze_stages >= 0:
            self.freeze_layers(freeze_stages)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * self._get_expansion(block):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self._get_expansion(block), kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * self._get_expansion(block)),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # Pass norm_layer if block takes it
        self.inplanes = planes * self._get_expansion(block)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes)) # Pass norm_layer if block takes it

        return nn.Sequential(*layers)

    def _get_expansion(self, block):
        # For a simple ResidualBlock as defined in blocks.py, expansion is 1
        # If using Bottleneck block, it would be 4.
        if hasattr(block, 'expansion'):
            return block.expansion
        return 1

    def forward(self, x):
        features = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['c1_pre_pool'] = x # Feature map before max pooling
        x = self.maxpool(x)
        features['c1'] = x # Output of layer0 or stem (e.g., 64 channels, H/4, W/4)

        x = self.layer1(x)
        features['c2'] = x # Output of layer1 (e.g., 64/256 channels, H/4, W/4)

        x = self.layer2(x)
        features['c3'] = x # Output of layer2 (e.g., 128/512 channels, H/8, W/8)

        x = self.layer3(x)
        features['c4'] = x # Output of layer3 (e.g., 256/1024 channels, H/16, W/16)

        x = self.layer4(x)
        features['c5'] = x # Output of layer4 (e.g., 512/2048 channels, H/32, W/32)

        return features

# Example usage:
# resnet_backbone = ResNetBackbone(layers=[2, 2, 2, 2]) # A smaller ResNet, e.g., ResNet-18 like
# dummy_input = torch.randn(1, 3, 224, 224)
# feats = resnet_backbone(dummy_input)
# for name, feat in feats.items():
#     print(f"{name}: {feat.shape}")

# Output for ResNet-18 like with input (1, 3, 224, 224):
# c1_pre_pool: torch.Size([1, 64, 112, 112])
# c1: torch.Size([1, 64, 56, 56])
# c2: torch.Size([1, 64, 56, 56])
# c3: torch.Size([1, 128, 28, 28])
# c4: torch.Size([1, 256, 14, 14])
# c5: torch.Size([1, 512, 7, 7])

def get_resnet50_backbone(input_channels=3, pretrained=False):
    """
    Returns a ResNet50-like backbone.
    If pretrained is True, this function should load weights from a .pth file or similar.
    For now, it just initializes the model.
    """
    model = ResNetBackbone(ResidualBlock, [3, 4, 6, 3], in_channels=input_channels)
    # if pretrained:
    #     # Placeholder for loading pretrained weights
    #     print("Loading pretrained weights for ResNet50 backbone (not implemented yet).")
    #     # state_dict = torch.load(PATH_TO_PRETRAINED_WEIGHTS)
    #     # model.load_state_dict(state_dict)
    return model

def get_resnet18_backbone(input_channels=3, pretrained=False):
    model = ResNetBackbone(ResidualBlock, [2, 2, 2, 2], in_channels=input_channels)
    return model

if __name__ == '__main__':
    # Test with ResNet-18 like structure
    print("Testing ResNet-18 like backbone:")
    resnet_backbone_18 = get_resnet18_backbone(input_channels=3)
    dummy_input_18 = torch.randn(1, 3, 256, 256) # Common size for segmentation
    feats_18 = resnet_backbone_18(dummy_input_18)
    for name, feat in feats_18.items():
        print(f"{name}: {feat.shape}")
    # Expected output for ResNet-18 like with input (1, 3, 256, 256):
    # c1_pre_pool: torch.Size([1, 64, 128, 128])
    # c1: torch.Size([1, 64, 64, 64])  (after maxpool)
    # c2: torch.Size([1, 64, 64, 64])  (layer1 output)
    # c3: torch.Size([1, 128, 32, 32]) (layer2 output)
    # c4: torch.Size([1, 256, 16, 16]) (layer3 output)
    # c5: torch.Size([1, 512, 8, 8])   (layer4 output, this is the one fed to ViT)

    print("\nTesting ResNet-50 like backbone:")
    resnet_backbone_50 = get_resnet50_backbone(input_channels=3)
    dummy_input_50 = torch.randn(1, 3, 256, 256)
    feats_50 = resnet_backbone_50(dummy_input_50)
    for name, feat in feats_50.items():
        print(f"{name}: {feat.shape}")
    # Expected output for ResNet-50 like with input (1, 3, 256, 256):
    # c1_pre_pool: torch.Size([1, 64, 128, 128])
    # c1: torch.Size([1, 64, 64, 64])
    # c2: torch.Size([1, 64, 64, 64])  (layer1 output, using BasicBlock, channels would be 256 if Bottleneck)
    # c3: torch.Size([1, 128, 32, 32]) (layer2 output, using BasicBlock, channels would be 512 if Bottleneck)
    # c4: torch.Size([1, 256, 16, 16]) (layer3 output, using BasicBlock, channels would be 1024 if Bottleneck)
    # c5: torch.Size([1, 512, 8, 8])   (layer4 output, using BasicBlock, channels would be 2048 if Bottleneck)
    # Note: The channel numbers for c2-c5 are based on the BasicBlock (ResidualBlock).
    # If Bottleneck blocks were used (as in standard ResNet50), these would be x4.
    # For TransUNet, the output of layer4 (c5) is typically used.
    # The standard TransUNet paper uses a ResNet50 backbone where the final feature map fed to ViT
    # has dimensions B x 2048 x H/32 x W/32.
    # My current ResidualBlock is a BasicBlock. To match ResNet50, I'd need to implement Bottleneck.
    # For now, this BasicBlock based ResNet is a placeholder.
    # The important part is getting features at different scales.
    # The TransUNet paper mentions using feature maps from R50-ViT-B_16:
    # The encoder path is based on ResNet-50, and the ViT uses ViT-B_16.
    # The output of ResNet-50 stage 4 (before avg pool) is 2048 channels.
    # This is then projected to ViT's hidden size (e.g., 768).
    # The skip connections are taken from ResNet-50 stages 1, 2, 3.
    # My 'c2' corresponds to ResNet stage 1 (after initial stem).
    # My 'c3' corresponds to ResNet stage 2.
    # My 'c4' corresponds to ResNet stage 3.
    # My 'c5' corresponds to ResNet stage 4.
    # So, for skip connections, we'd use features['c2'], features['c3'], features['c4'].
    # And features['c5'] would be fed into the ViT.

    # Let's adjust the ResNetBackbone to use Bottleneck for a ResNet50 like structure
    # For simplicity in this step, I'll stick with the current ResidualBlock (BasicBlock).
    # The channel dimensions will be smaller than a standard ResNet50, but the structure is there.
    # We can refine this (e.g., add Bottleneck) if performance is impacted significantly or for closer replication.
    # The key is the multi-scale feature extraction.

    def load_pretrained_weights(self, path):
        try:
            state_dict = torch.load(path, map_location='cpu')
            # Handle potential 'state_dict' key if saved from a LightningModule or similar wrapper
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Basic filtering: remove keys not part of this model's state_dict
            # More sophisticated key mapping might be needed if names differ significantly
            # (e.g. "backbone.conv1.weight" vs "conv1.weight")
            model_state_dict = self.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                # Attempt to match keys, removing "module." prefix if present (from DataParallel saving)
                new_k = k.replace("module.", "")
                if new_k in model_state_dict and model_state_dict[new_k].shape == v.shape:
                    filtered_state_dict[new_k] = v
                # elif any(new_k.endswith(model_k) for model_k in model_state_dict.keys()):
                #     # Try to find if it's a sub-module's weight, e.g. backbone.layer1...
                #     pass # More complex mapping needed here

            if not filtered_state_dict:
                logger.warning(f"No matching keys found in pretrained weights at {path} for ResNetBackbone.")
                return

            missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"ResNetBackbone: Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"ResNetBackbone: Unexpected keys when loading pretrained weights: {unexpected_keys}")
            logger.info(f"ResNetBackbone: Successfully loaded {len(filtered_state_dict)} matching parameters from {path}")

        except Exception as e:
            logger.error(f"Error loading pretrained weights for ResNetBackbone from {path}: {e}")

    def freeze_layers(self, freeze_stages):
        """
        Freezes layers of the ResNet backbone.
        freeze_stages:
            -1: Unfreeze all.
             0: Freeze conv1, bn1.
             1: Freeze stage 0 and layer1 (c1, c2 features).
             2: Freeze stages 0, 1, and layer2 (c1, c2, c3 features).
             3: Freeze stages 0, 1, 2, and layer3 (c1, c2, c3, c4 features).
             4: Freeze all stages (entire backbone).
        """
        if freeze_stages >= 0:
            self.conv1.requires_grad_(False)
            self.bn1.requires_grad_(False)
            for param in self.conv1.parameters(): param.requires_grad = False
            for param in self.bn1.parameters(): param.requires_grad = False
            logger.info("ResNetBackbone: Froze initial conv1 and bn1.")

        if freeze_stages >= 1:
            for param in self.layer1.parameters(): param.requires_grad = False
            logger.info("ResNetBackbone: Froze layer1.")
        else: # Unfreeze if freeze_stages < 1 (but >=0 handled above)
            for param in self.layer1.parameters(): param.requires_grad = True


        if freeze_stages >= 2:
            for param in self.layer2.parameters(): param.requires_grad = False
            logger.info("ResNetBackbone: Froze layer2.")
        else:
            for param in self.layer2.parameters(): param.requires_grad = True

        if freeze_stages >= 3:
            for param in self.layer3.parameters(): param.requires_grad = False
            logger.info("ResNetBackbone: Froze layer3.")
        else:
            for param in self.layer3.parameters(): param.requires_grad = True

        if freeze_stages >= 4:
            for param in self.layer4.parameters(): param.requires_grad = False
            logger.info("ResNetBackbone: Froze layer4.")
        else:
            for param in self.layer4.parameters(): param.requires_grad = True
    pass
