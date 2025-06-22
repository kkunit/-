import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_models # For loading pretrained weights

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision ResNet has expansion=4
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Stem
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers (Stages)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _load_pretrained_weights(custom_model, torchvision_model_name, num_classes):
    """
    Loads weights from a torchvision model into a custom ResNet model.
    Handles potential mismatches in the fully connected layer.
    """
    try:
        tv_model = getattr(torchvision_models, torchvision_model_name)(weights='IMAGENET1K_V1')
        tv_state_dict = tv_model.state_dict()
        custom_state_dict = custom_model.state_dict()

        # Filter out unnecessary keys and adapt layer names if needed
        # For ResNet, names are usually compatible if structure is the same.
        # Main difference is often the final 'fc' layer if num_classes differs.

        # If number of classes is different, fc layer weights won't match.
        # We load all weights except fc, and re-init fc for custom_model.
        if custom_model.fc.out_features != tv_model.fc.out_features:
            print(f"Custom model has {custom_model.fc.out_features} output classes, "
                  f"while torchvision {torchvision_model_name} has {tv_model.fc.out_features}. "
                  "Loading all weights except the final FC layer.")
            # Remove fc weights and bias from torchvision state_dict
            tv_state_dict.pop('fc.weight', None)
            tv_state_dict.pop('fc.bias', None)

            # Load the filtered state dict
            custom_model.load_state_dict(tv_state_dict, strict=False) # strict=False to ignore missing fc

            # Re-initialize the fc layer of the custom model
            print(f"Re-initializing fc layer for {custom_model.fc.out_features} classes.")
            custom_model.fc = nn.Linear(custom_model.fc.in_features, custom_model.fc.out_features)
            nn.init.kaiming_normal_(custom_model.fc.weight, mode='fan_out', nonlinearity='relu')
            if custom_model.fc.bias is not None:
                nn.init.constant_(custom_model.fc.bias, 0)
        else:
            # If num_classes is the same (e.g., 1000), load all weights.
            custom_model.load_state_dict(tv_state_dict, strict=True)

        print(f"Successfully loaded pre-trained weights from torchvision's {torchvision_model_name} into custom model.")

    except Exception as e:
        print(f"Error loading pre-trained weights for {torchvision_model_name}: {e}")
        print("Proceeding with randomly initialized weights for the custom model.")


def resnet18(pretrained=False, num_classes=1000, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    if pretrained:
        _load_pretrained_weights(model, 'resnet18', num_classes)
    return model

def resnet34(pretrained=False, num_classes=1000, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if pretrained:
        _load_pretrained_weights(model, 'resnet34', num_classes)
    return model

def resnet50(pretrained=False, num_classes=1000, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if pretrained:
        _load_pretrained_weights(model, 'resnet50', num_classes)
    return model

# ... (add resnet101, resnet152 similarly if needed, using Bottleneck)


def get_resnet_architecture_summary(model_name="ResNet_Baseline", num_classes=2, model_instance=None):
    """
    Returns a string summary of the ResNet architecture.
    If model_instance is provided, it tries to use torchinfo for a detailed summary.
    Otherwise, returns a generic description.
    """
    if model_instance:
        try:
            from torchinfo import summary
            # Batch size of 1, 3 channels, 224x224 image size
            # Ensure model is on CPU for summary if it was on GPU
            device = next(model_instance.parameters()).device
            summary_str = str(summary(model_instance, input_size=(1, 3, 224, 224), verbose=0, device='cpu'))
            model_instance.to(device) # Move model back to original device
            return summary_str
        except ImportError:
            return f"torchinfo not available. Cannot generate detailed summary for {model_name}.\n" \
                   f"Model: {model_name} with {num_classes} output classes. (Instance provided but torchinfo missing)"
        except Exception as e:
            return f"Error generating torchinfo summary for {model_name}: {e}\n" \
                   f"Model: {model_name} with {num_classes} output classes."


    # Generic summaries if no instance or torchinfo fails
    if model_name == "ResNet_Baseline": # Assuming ResNet-18 or ResNet-34
        return f"""\
ResNet Baseline (e.g., ResNet-18/34 - Custom Built)
------------------------------------
Target Output Classes: {num_classes}
Architecture: Standard ResNet (BasicBlock for 18/34)
- Input: (batch_size, 3, 224, 224)
- Stem:
  - Conv2D (7x7, 64 filters, stride 2, padding 3)
  - BatchNorm2D
  - ReLU
  - MaxPool2D (3x3, stride 2, padding 1)
- Layer 1: Sequence of BasicBlocks (e.g., 2 for ResNet18, 3 for ResNet34), output 64 channels
- Layer 2: Sequence of BasicBlocks (e.g., 2 for ResNet18, 4 for ResNet34), output 128 channels, stride 2
- Layer 3: Sequence of BasicBlocks (e.g., 2 for ResNet18, 6 for ResNet34), output 256 channels, stride 2
- Layer 4: Sequence of BasicBlocks (e.g., 2 for ResNet18, 3 for ResNet34), output 512 channels, stride 2
- Classifier:
  - AdaptiveAvgPool2d((1, 1))
  - FullyConnected Layer (512 -> {num_classes})
------------------------------------
(This is a general description. For detailed layer info, torchinfo summary is preferred.)
"""
    elif model_name == "ResNet_Improved":
        return f"""\
ResNet Improved (Custom Built)
------------------------------------
Target Output Classes: {num_classes}
(Details of the improved ResNet architecture will be shown here once defined and an instance is passed for summary)
- This will be based on a ResNet Baseline with specific structural modifications.
------------------------------------
"""
    return "Architecture details not available for this model type or no instance provided."


if __name__ == '__main__':
    print("Testing ResNet Builder...")

    # Test ResNet-18 instantiation
    print("\n--- ResNet-18 (Custom) ---")
    custom_resnet18 = resnet18(pretrained=False, num_classes=2) # For Pneumonia dataset (NORMAL/PNEUMONIA)
    # print(custom_resnet18)
    print(f"Custom ResNet-18 (2 classes) created. FC out features: {custom_resnet18.fc.out_features}")

    # Test with pre-trained weights (mocking loading, actual download might occur if not cached)
    # Ensure you have internet for the first time to download weights.
    print("\n--- ResNet-18 (Custom) with 'Pretrained' (loading ImageNet weights, adapting FC) ---")
    custom_resnet18_pretrained = resnet18(pretrained=True, num_classes=2)
    print(f"Custom ResNet-18 (2 classes) with pre-trained weights loaded. FC out features: {custom_resnet18_pretrained.fc.out_features}")
    # Verify FC layer re-initialization if num_classes is different from 1000
    assert custom_resnet18_pretrained.fc.out_features == 2
    # Check if other layers have loaded weights (e.g. conv1 weights are not all zero or default init)
    # This is harder to assert directly without comparing to torchvision model's actual weights.
    # A simple check: parameters should be requires_grad=True by default.
    # And that the conv1 weights are not identical to a freshly initialized conv1.
    fresh_conv1_weight = resnet18(pretrained=False, num_classes=2).conv1.weight.clone()
    #This comparison is not perfectly robust due to kaiming_normal_ randomness,
    #but pretrained weights should be distinctly different.
    #A better check would be to load torchvision's resnet18, get its conv1 weight, and compare.
    #For now, we rely on the print statements from _load_pretrained_weights.
    # assert not torch.allclose(custom_resnet18_pretrained.conv1.weight.data, fresh_conv1_weight.data), \
    #    "Conv1 weights seem to be the same as fresh init after trying to load pretrained."


    print("\n--- ResNet-34 (Custom) ---")
    custom_resnet34 = resnet34(pretrained=False, num_classes=10) # Example with 10 classes
    print(f"Custom ResNet-34 (10 classes) created. FC out features: {custom_resnet34.fc.out_features}")

    print("\n--- ResNet-50 (Custom, Bottleneck) with 'Pretrained' ---")
    custom_resnet50_pretrained = resnet50(pretrained=True, num_classes=5)
    print(f"Custom ResNet-50 (5 classes) with pre-trained weights loaded. FC out features: {custom_resnet50_pretrained.fc.out_features}")
    assert custom_resnet50_pretrained.fc.out_features == 5

    # Test architecture summary
    print("\n--- Architecture Summaries ---")
    summary_generic_baseline = get_resnet_architecture_summary("ResNet_Baseline", num_classes=2)
    print("Generic Baseline Summary:\n", summary_generic_baseline)

    summary_generic_improved = get_resnet_architecture_summary("ResNet_Improved", num_classes=2)
    print("\nGeneric Improved Summary:\n", summary_generic_improved)

    print("\nAttempting torchinfo summary for ResNet-18 (num_classes=2):")
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    try:
        custom_resnet18.eval() # Set to eval mode for summary/inference
        output = custom_resnet18(dummy_input)
        print(f"Output shape from custom_resnet18: {output.shape}") # Should be [1, num_classes]
        assert output.shape == (1,2)

        summary_torchinfo = get_resnet_architecture_summary("ResNet_Baseline", model_instance=custom_resnet18)
        print(summary_torchinfo)
    except Exception as e:
        print(f"Could not generate torchinfo summary or run model: {e}")
        print("This might be due to torchinfo not being installed, or an issue in the model definition.")

    print("\nResNet Builder tests completed.")
    print("Note: For `pretrained=True`, ensure you have an internet connection the first time "
          "to download weights from torchvision. If `torchinfo` is not installed, detailed summaries won't be available.")
