import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        out = nn.ReLU(inplace=True)(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, input_channels=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Modified for 1-channel input and small images (28x28)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False) # Adjusted kernel_size, stride, padding
        self.bn1 = nn.BatchNorm2d(64)
        # Removed MaxPool2d for small images: self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.in_planes, planes, stride_val))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        # out = self.pool1(out) # Removed MaxPool2d
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet50_custom(num_classes=2, input_channels=1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, input_channels=input_channels)

if __name__ == '__main__':
    # Test the model definition
    print("Defining ResNet50 for 1-channel 28x28 images and 2 output classes.")
    model = ResNet50_custom(num_classes=2, input_channels=1)
    print(model)

    # Verify input and output layers
    print("\nVerifying model structure:")
    print(f"Input conv1: {model.conv1}")
    print(f"Output fc: {model.fc}")

    # Test with a dummy input
    try:
        dummy_input = torch.randn(1, 1, 28, 28) # (batch_size, channels, height, width)
        output = model(dummy_input)
        print(f"\nDummy input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        if output.shape == torch.Size([1, 2]):
            print("Model test with dummy input successful. Output shape is correct.")
        else:
            print(f"Model test with dummy input FAILED. Output shape is {output.shape}, expected torch.Size([1, 2]).")

    except Exception as e:
        print(f"\nError during model test with dummy input: {e}")
