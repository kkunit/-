import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .cnn_backbone import get_resnet50_backbone, get_resnet18_backbone, ResNetBackbone
from .vision_transformer import VisionTransformer
from .blocks import ConvBlock, UpsampleBlock, AttentionGate # Added AttentionGate

# Default configuration for ViT-B_16 used in TransUNet
VIT_CONFIG = {
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4.0,
    'dropout': 0.0, # Usually 0.0 for inference/pre-training, 0.1 for fine-tuning
}

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        # Attention Gate for the skip connection
        # F_g: channels of the gating signal (x_upsampled, which has 'in_channels')
        # F_l: channels of the skip connection features ('skip_channels')
        # F_int: intermediate channels in AG, typically F_l // 2 or max(1, F_l // 2)
        self.attn_gate = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=max(1, skip_channels // 2))

        # Convolutional blocks after concatenation
        # The number of input channels to conv1 remains in_channels (from x_upsampled) + skip_channels (from skip_att)
        # because AttentionGate only re-weights skip_channels, does not change its channel count.
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_feature):
        # x is from previous decoder layer (more coarse)
        # skip_feature is from corresponding encoder layer (more fine)

        x_upsampled = self.upsample(x) # Upsample x to increase resolution

        # Spatial alignment: Ensure x_upsampled matches skip_feature's spatial dimensions.
        # This is critical before AttentionGate and before concatenation.
        if x_upsampled.shape[2:] != skip_feature.shape[2:]:
            x_upsampled = F.interpolate(x_upsampled, size=skip_feature.shape[2:], mode='bilinear', align_corners=True)

        # Apply AttentionGate to the skip_feature, using x_upsampled as the gating signal
        skip_feature_att = self.attn_gate(g=x_upsampled, x=skip_feature)

        # Concatenate the upsampled features and the attention-gated skip features
        x_concat = torch.cat([x_upsampled, skip_feature_att], dim=1)

        # Apply convolutional blocks
        x_out = self.conv1(x_concat)
        x_out = self.conv2(x_out)
        return x_out

class TransUNet(nn.Module):
    def __init__(self, img_size=(256, 256), num_classes=1,
                 cnn_backbone_type='resnet50', # 'resnet18', 'resnet50'
                 cnn_in_channels=3,
                 cnn_pretrained_weights_path=None,
                 cnn_freeze_stages=-1,
                 vit_patch_size=1, # Patch size for ViT input (feature map from CNN)
                 vit_config=None,
                 vit_pretrained_weights_path=None,
                 vit_freeze_patch_embed=False,
                 vit_freeze_pos_embed=False,
                 vit_freeze_blocks_upto=-1,
                 decoder_channels=(256, 128, 64, 16), # Channels for decoder stages
                 final_upsample_factor=4 # To restore to original image size from the last CNN skip connection
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.vit_config = vit_config if vit_config is not None else VIT_CONFIG.copy()

        # 1. CNN Backbone
        if cnn_backbone_type == 'resnet18':
            self.cnn_backbone = get_resnet18_backbone(
                input_channels=cnn_in_channels,
                pretrained_weights_path=cnn_pretrained_weights_path,
                freeze_stages=cnn_freeze_stages
            )
            # For ResNet18 (BasicBlock based):
            # c2: 64 (H/4, W/4) - skip1
            # c3: 128 (H/8, W/8) - skip2
            # c4: 256 (H/16, W/16) - skip3
            # c5: 512 (H/32, W/32) - to ViT
            cnn_feature_channels = {'skip1': 64, 'skip2': 128, 'skip3': 256, 'to_vit': 512}
            self.vit_input_hw = (img_size[0] // 32, img_size[1] // 32)
        elif cnn_backbone_type == 'resnet50':
            # If using a true ResNet50 with Bottleneck blocks, channels would be:
            # c2: 256 (layer1 output)
            # c3: 512 (layer2 output)
            # c4: 1024 (layer3 output)
            # c5: 2048 (layer4 output)
            # For now, our ResNetBackbone uses BasicBlock, so it's same as ResNet18 channels.
            # This needs to be adjusted if a Bottleneck ResNet50 is implemented.
            # For demonstration, let's assume we use the BasicBlock version for now.
            self.cnn_backbone = get_resnet50_backbone(input_channels=cnn_in_channels)
            cnn_feature_channels = {'skip1': 64, 'skip2': 128, 'skip3': 256, 'to_vit': 512} # BasicBlock based
            # If using Bottleneck ResNet50, these would be:
            # cnn_feature_channels = {'skip1': 256, 'skip2': 512, 'skip3': 1024, 'to_vit': 2048}
            self.vit_input_hw = (img_size[0] // 32, img_size[1] // 32)
        elif cnn_backbone_type == 'resnet50':
            self.cnn_backbone = get_resnet50_backbone(
                input_channels=cnn_in_channels,
                pretrained_weights_path=cnn_pretrained_weights_path,
                freeze_stages=cnn_freeze_stages
            )
            # Assuming BasicBlock version for channels, adjust if Bottleneck is used
            cnn_feature_channels = {'skip1': 64, 'skip2': 128, 'skip3': 256, 'to_vit': 512}
            self.vit_input_hw = (img_size[0] // 32, img_size[1] // 32)
        else:
            raise ValueError(f"Unsupported CNN backbone type: {cnn_backbone_type}")

        # 2. Vision Transformer Encoder
        self.vit = VisionTransformer(
            img_size=self.vit_input_hw, # H/32, W/32
            patch_size=vit_patch_size, # Usually 1 for TransUNet
            in_channels=cnn_feature_channels['to_vit'],
            embed_dim=self.vit_config['embed_dim'],
            depth=self.vit_config['depth'],
            num_heads=self.vit_config['num_heads'],
            mlp_ratio=self.vit_config['mlp_ratio'],
            dropout=self.vit_config['dropout'],
            num_classes=0, # No classification head
            pretrained_weights_path=vit_pretrained_weights_path,
            freeze_patch_embed=vit_freeze_patch_embed,
            freeze_pos_embed=vit_freeze_pos_embed,
            freeze_blocks_upto=vit_freeze_blocks_upto
        )

        # 3. Decoder (Cascaded Upsampler - CUP)
        # The ViT output needs to be reshaped back to 2D feature map
        # ViT output: (B, num_patches, embed_dim) where num_patches = (H/32)*(W/32)
        # Reshape to: (B, embed_dim, H/32, W/32)
        # This is the input to the first decoder block.

        # Decoder stage 1: Upsample ViT output and combine with c4 (skip3)
        # Input to decoder1 is reshaped ViT output.
        # Skip feature is c4 (e.g., 256 channels for BasicBlock ResNet, H/16, W/16)
        self.decoder1 = DecoderBlock(
            in_channels=self.vit_config['embed_dim'],
            skip_channels=cnn_feature_channels['skip3'],
            out_channels=decoder_channels[0] # e.g., 256
        ) # Output: (B, dec_ch[0], H/16, W/16)

        # Decoder stage 2: Upsample decoder1 output and combine with c3 (skip2)
        # Skip feature is c3 (e.g., 128 channels for BasicBlock ResNet, H/8, W/8)
        self.decoder2 = DecoderBlock(
            in_channels=decoder_channels[0],
            skip_channels=cnn_feature_channels['skip2'],
            out_channels=decoder_channels[1] # e.g., 128
        ) # Output: (B, dec_ch[1], H/8, W/8)

        # Decoder stage 3: Upsample decoder2 output and combine with c2 (skip1)
        # Skip feature is c2 (e.g., 64 channels for BasicBlock ResNet, H/4, W/4)
        self.decoder3 = DecoderBlock(
            in_channels=decoder_channels[1],
            skip_channels=cnn_feature_channels['skip1'],
            out_channels=decoder_channels[2] # e.g., 64
        ) # Output: (B, dec_ch[2], H/4, W/4)

        # Decoder stage 4: Upsample decoder3 output to a higher resolution
        # No skip connection here from CNN stem directly, but could be from c1_pre_pool or similar
        # TransUNet paper description: "The final up-sampling layer further up-samples the feature maps by 4x"
        # This implies the output of decoder3 (H/4, W/4) is upsampled to (H, W)
        # The number of channels is typically reduced further.
        self.decoder4_upsample = nn.Upsample(scale_factor=final_upsample_factor, mode='bilinear', align_corners=True)
        self.decoder4_conv = ConvBlock(decoder_channels[2], decoder_channels[3], kernel_size=3, padding=1) # e.g., 64 -> 16

        # Final segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, C_in, H, W)

        # 1. CNN Backbone feature extraction
        cnn_features = self.cnn_backbone(x)
        # cnn_features is a dict: {'c1': ..., 'c2': ..., 'c3': ..., 'c4': ..., 'c5': ...}
        # Based on ResNetBackbone:
        # skip1 = cnn_features['c2'] # H/4, W/4 (e.g., 64 ch for BasicBlock ResNet)
        # skip2 = cnn_features['c3'] # H/8, W/8 (e.g., 128 ch)
        # skip3 = cnn_features['c4'] # H/16, W/16 (e.g., 256 ch)
        # vit_input_map = cnn_features['c5'] # H/32, W/32 (e.g., 512 ch)

        skip1, skip2, skip3 = cnn_features['c2'], cnn_features['c3'], cnn_features['c4']
        vit_input_map = cnn_features['c5']

        # 2. Vision Transformer Encoder
        # vit_input_map: (B, C_vit_in, H_vit_in, W_vit_in) = (B, 512, H/32, W/32)
        vit_output = self.vit(vit_input_map) # (B, num_patches, embed_dim)

        # Reshape ViT output to 2D feature map for decoder
        # num_patches = (H/32) * (W/32)
        # H_vit_in, W_vit_in = self.vit_input_hw
        B, N, E = vit_output.shape # Batch, NumPatches, EmbedDim
        H_vit_in, W_vit_in = self.vit_input_hw[0], self.vit_input_hw[1] # Should match vit_input_map.shape[2:]

        # Ensure N matches H_vit_in * W_vit_in
        if N != H_vit_in * W_vit_in:
             raise ValueError(f"ViT output sequence length {N} does not match expected spatial dimensions {H_vit_in}*{W_vit_in}={H_vit_in * W_vit_in}")

        x_decoder_input = vit_output.transpose(1, 2).contiguous().view(B, E, H_vit_in, W_vit_in)
        # x_decoder_input: (B, embed_dim, H/32, W/32)

        # 3. Decoder
        d1 = self.decoder1(x_decoder_input, skip3) # Output: (B, dec_ch[0], H/16, W/16)
        d2 = self.decoder2(d1, skip2)             # Output: (B, dec_ch[1], H/8, W/8)
        d3 = self.decoder3(d2, skip1)             # Output: (B, dec_ch[2], H/4, W/4)

        d4 = self.decoder4_upsample(d3)           # Output: (B, dec_ch[2], H, W)
        d4 = self.decoder4_conv(d4)               # Output: (B, dec_ch[3], H, W)

        # 4. Segmentation Head
        logits = self.seg_head(d4)                # Output: (B, num_classes, H, W)

        # Apply sigmoid for binary or multi-label, or softmax for multi-class if needed (often done in loss)
        # For binary segmentation (num_classes=1), sigmoid is common.
        # If num_classes > 1 and it's multi-class, softmax along channel dim.
        # For now, return raw logits.

        return logits

# Example Usage
if __name__ == '__main__':
    img_size = (224, 224) # Standard ViT size, but TransUNet often uses 256x256 or 512x512
    num_classes = 1 # Binary segmentation (e.g., foreground/background)

    # Test with ResNet18-like backbone
    print("Testing TransUNet with ResNet18-like backbone:")
    transunet_r18 = TransUNet(
        img_size=img_size,
        num_classes=num_classes,
        cnn_backbone_type='resnet18',
        cnn_in_channels=3,
        # vit_config can be customized here if needed, e.g., smaller ViT
    )
    dummy_input = torch.randn(2, 3, img_size[0], img_size[1]) # Batch size 2
    output_r18 = transunet_r18(dummy_input)
    print(f"Output shape (ResNet18 backbone): {output_r18.shape}") # Expected: (2, num_classes, img_size[0], img_size[1])
    assert output_r18.shape == (2, num_classes, img_size[0], img_size[1])

    num_params_r18 = sum(p.numel() for p in transunet_r18.parameters() if p.requires_grad)
    print(f"TransUNet (ResNet18) learnable parameters: {num_params_r18 / 1e6:.2f} M")


    # Test with ResNet50-like backbone (using BasicBlocks for now)
    # To truly match TransUNet paper, ResNet50 with Bottleneck blocks would be needed,
    # and the cnn_feature_channels and vit_in_channels would change.
    # For now, this demonstrates the structure.
    print("\nTesting TransUNet with ResNet50-like (BasicBlock) backbone:")
    # A smaller ViT config for faster testing with ResNet50-like
    small_vit_config = {
        'embed_dim': 512, # Smaller than ViT-Base 768
        'depth': 8,       # Smaller than ViT-Base 12
        'num_heads': 8,   # Smaller than ViT-Base 12
        'mlp_ratio': 3.0,
        'dropout': 0.1,
    }
    transunet_r50_custom_vit = TransUNet(
        img_size=img_size,
        num_classes=num_classes,
        cnn_backbone_type='resnet50', # Still uses BasicBlock ResNet50 for now
        cnn_in_channels=3,
        vit_config=small_vit_config,
        decoder_channels=(256, 128, 64, 32) # Adjusted decoder channels
    )
    output_r50_custom = transunet_r50_custom_vit(dummy_input)
    print(f"Output shape (ResNet50 backbone, custom ViT): {output_r50_custom.shape}")
    assert output_r50_custom.shape == (2, num_classes, img_size[0], img_size[1])

    num_params_r50_custom = sum(p.numel() for p in transunet_r50_custom_vit.parameters() if p.requires_grad)
    print(f"TransUNet (ResNet50, custom ViT) learnable parameters: {num_params_r50_custom / 1e6:.2f} M")

    # A key detail from TransUNet paper:
    # "The skip-connections link the encoder path of ResNet-50 and the decoder path of CUP.
    # Specifically, feature maps from stages 1, 2, and 3 of ResNet-50 are connected to corresponding
    # up-sampling stages in CUP. These feature maps have resolutions of H/4 × W/4, H/8 × W/8, and H/16 × W/16,
    # respectively. The channels of these feature maps are 256, 512, and 1024 for R50-ViT-B_16."
    # This means my `cnn_feature_channels` for a true ResNet50 backbone should be:
    # skip1: 256 (from ResNet layer1 output, which is after conv2_x)
    # skip2: 512 (from ResNet layer2 output, which is after conv3_x)
    # skip3: 1024 (from ResNet layer3 output, which is after conv4_x)
    # to_vit: 2048 (from ResNet layer4 output, which is after conv5_x)
    # My current `ResNetBackbone` with `ResidualBlock` (BasicBlock) does not produce these channel numbers.
    # To fix this for a "true" ResNet50 backbone, `ResNetBackbone` would need to use Bottleneck blocks,
    # and the `_get_expansion` method would return 4.
    #
    # For the current implementation using BasicBlocks, the channel numbers are smaller, but the
    # general architecture (CNN -> ViT -> Decoder with skips) is in place.
    # The `final_upsample_factor` ensures the output matches input spatial dims.
    # If the last skip is from H/4, W/4, then factor=4 is correct.
    # If `img_size` is not divisible by 32 (max downsampling factor), there might be slight size mismatches
    # in skip connections. F.interpolate can be used inside DecoderBlock if necessary.
    # The current `ResNetBackbone` ensures divisibility by using strides of 2.

    # A note on ViT input size:
    # For an input image of 224x224, H/32 = 7, W/32 = 7. So vit_input_hw = (7,7).
    # Number of patches for ViT = 7*7 = 49.
    # The `VisionTransformer` `img_size` parameter should match this.
    # My code dynamically calculates `self.vit_input_hw` based on `img_size` and downsampling factor.
    if img_size == (224,224):
        assert transunet_r18.vit_input_hw == (7,7)
        assert transunet_r18.vit.num_patches == 49
        assert transunet_r18.vit.pos_embed.shape == (1, 49, transunet_r18.vit_config['embed_dim'])

    print("Basic TransUNet structure implemented.")
