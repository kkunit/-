import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import logging

logger = logging.getLogger(__name__)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size # e.g., (H, W) of the feature map from CNN
        self.patch_size = patch_size # e.g., 1 for TransUNet if input is already patched
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # If patch_size is 1, it means the input feature map is already "patched"
        # The projection is then just a linear projection of the channel dimension
        if patch_size == 1:
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1)
        else:
            # This case is more like standard ViT, not directly used in TransUNet's ViT part for 1x1 patches
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x is (B, C, H, W) from CNN backbone
        x = self.proj(x)  # (B, embed_dim, H', W') where H'=H/patch_size, W'=W/patch_size
        # Flatten H' and W' into a sequence
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape # Batch, Num_patches, Channels (embed_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # qkv shape: (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # each (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn # Returning attention weights can be useful for visualization/analysis

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        h, att_weights = self.attn(self.norm1(x))
        x = x + h # Residual connection
        x = x + self.mlp(self.norm2(x)) # Residual connection
        return x, att_weights

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_ratio=4.0, dropout=0.1, num_classes=1000,
                 pretrained_weights_path=None, freeze_patch_embed=False, freeze_pos_embed=False, freeze_blocks_upto=-1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Positional embedding
        # TransUNet uses learnable 1D positional embeddings.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head (not used directly in TransUNet encoder part, but standard for ViT)
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights) # Apply custom _init_weights

        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path) # Load weights if path provided

        # Apply freezing based on parameters
        self.freeze_layers(freeze_patch_embed, freeze_pos_embed, freeze_blocks_upto)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d): # For patch_embed.proj
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu') # Or trunc_normal_
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def load_pretrained_weights(self, path, strict_load=False):
        try:
            state_dict = torch.load(path, map_location='cpu')
            # Common practice: ViT weights might be under 'model', 'state_dict' or 'teacher' key from other repos
            if 'model' in state_dict: state_dict = state_dict['model']
            elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            elif 'teacher' in state_dict: state_dict = state_dict['teacher']


            # Key mapping for standard ViT pre-trained weights (e.g., from timm or Google's JAX ViT)
            # This is a common source of issues and might need adjustment based on the specific .pth file.
            # Example:
            # 'cls_token' -> not used in our TransUNet ViT encoder part directly unless added
            # 'pos_embed' -> 'pos_embed'
            # 'patch_embed.proj.weight' -> 'patch_embed.proj.weight'
            # 'patch_embed.proj.bias'   -> 'patch_embed.proj.bias'
            # 'blocks.0.norm1.weight'   -> 'blocks.0.norm1.weight'
            # 'blocks.0.attn.qkv.weight' -> 'blocks.0.attn.qkv.weight' (careful with fused vs separate q,k,v)
            # Our MultiHeadAttention has a single self.qkv linear layer. Many pre-trained ViTs also do.

            current_model_dict = self.state_dict()
            loaded_keys = []
            skipped_keys_shape = []
            skipped_keys_missing = []

            for k, v in state_dict.items():
                # Basic name cleaning (e.g. removing "module." prefix)
                new_k = k.replace("module.", "")

                # Specific key mappings if known differences exist:
                # e.g., if pretrained has 'transformer.encoder.layer.0...' and ours is 'blocks.0...'
                # new_k = new_k.replace('transformer.encoder.layer.', 'blocks.') # Example

                if new_k in current_model_dict:
                    if current_model_dict[new_k].shape == v.shape:
                        current_model_dict[new_k] = v
                        loaded_keys.append(new_k)
                    else:
                        skipped_keys_shape.append(f"{new_k} (Model: {current_model_dict[new_k].shape}, File: {v.shape})")
                else:
                    skipped_keys_missing.append(new_k)

            if not loaded_keys:
                logger.warning(f"ViT: No matching keys found in pretrained weights at {path}.")
                return

            self.load_state_dict(current_model_dict, strict=strict_load) # Use current_model_dict which has updated values

            logger.info(f"ViT: Successfully loaded {len(loaded_keys)} matching parameters from {path}")
            if skipped_keys_shape:
                logger.warning(f"ViT: Skipped due to shape mismatch: {skipped_keys_shape}")
            if skipped_keys_missing and strict_load: # Only warn if strict_load is true for missing keys not in model
                 logger.warning(f"ViT: Keys in pretrained not found in model: {skipped_keys_missing}")


        except Exception as e:
            logger.error(f"Error loading pretrained weights for VisionTransformer from {path}: {e}")

    def freeze_layers(self, freeze_patch_embed=True, freeze_pos_embed=True, freeze_blocks_upto=-1):
        """
        Freezes parts of the Vision Transformer.
        freeze_patch_embed: If True, freeze the patch embedding layer.
        freeze_pos_embed: If True, freeze the positional embedding.
        freeze_blocks_upto: Freeze Transformer blocks from index 0 up to this index (inclusive).
                            -1 means no blocks are frozen by this parameter.
        """
        if freeze_patch_embed:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            logger.info("ViT: Froze patch_embed.")
        else:
            for param in self.patch_embed.parameters():
                param.requires_grad = True

        if freeze_pos_embed:
            self.pos_embed.requires_grad = False
            logger.info("ViT: Froze pos_embed.")
        else:
            self.pos_embed.requires_grad = True

        for i, block in enumerate(self.blocks):
            if i <= freeze_blocks_upto:
                for param in block.parameters():
                    param.requires_grad = False
                if i == freeze_blocks_upto: # Log only once for the range
                     logger.info(f"ViT: Froze Transformer blocks 0 to {i}.")
            else:
                for param in block.parameters():
                    param.requires_grad = True
        if freeze_blocks_upto == -1:
            logger.info("ViT: All Transformer blocks are trainable (or subject to other freeze settings).")


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        att_weights_list = []
        for blk in self.blocks:
            x, att_weights = blk(x)
            att_weights_list.append(att_weights)

        x = self.norm(x)
        return x, att_weights_list # Return all attention weights for potential analysis

    def forward(self, x):
        # x is the feature map from CNN, e.g., (B, C_cnn, H_cnn, W_cnn)
        # In TransUNet, this x is typically (B, 512, H/32, W/32) if using ResNet BasicBlock based backbone
        # or (B, 2048, H/32, W/32) if using ResNet Bottleneck based backbone.
        # The PatchEmbedding will project C_cnn to embed_dim.
        x, _ = self.forward_features(x) # We only need the features, not attention weights here for TransUNet encoder
        # x is (B, num_patches, embed_dim)
        return x

# Example Usage (mimicking TransUNet scenario)
if __name__ == '__main__':
    # Assume input feature map from CNN is 8x8 with 512 channels (e.g., for a 256x256 input image, H/32 = 8)
    # This would be the 'c5' feature from our ResNetBackbone.
    # Batch size 1, 512 channels, 8x8 spatial dimensions
    dummy_cnn_feature_map = torch.randn(1, 512, 8, 8)

    # ViT parameters for TransUNet (example, these can be configured)
    # img_size for ViT is the H,W of the input feature map
    vit_img_size = (8, 8)
    # patch_size = 1 means each "pixel" in the 8x8 feature map is a patch.
    # This is how TransUNet applies ViT: it doesn't break the feature map into larger patches.
    vit_patch_size = 1
    vit_in_channels = 512 # Channels from CNN feature map
    vit_embed_dim = 768   # Standard ViT-Base hidden size
    vit_depth = 12        # Standard ViT-Base depth
    vit_num_heads = 12    # Standard ViT-Base heads
    vit_mlp_ratio = 4.0
    vit_dropout = 0.0 # Usually 0.0 for inference, 0.1 for training

    vit_encoder = VisionTransformer(
        img_size=vit_img_size,
        patch_size=vit_patch_size,
        in_channels=vit_in_channels,
        embed_dim=vit_embed_dim,
        depth=vit_depth,
        num_heads=vit_num_heads,
        mlp_ratio=vit_mlp_ratio,
        dropout=vit_dropout,
        num_classes=0 # No classifier head needed for TransUNet encoder part
    )

    output_features = vit_encoder(dummy_cnn_feature_map)
    # Expected output shape: (B, num_patches, embed_dim)
    # num_patches = (8/1) * (8/1) = 64
    # So, shape should be (1, 64, 768)
    print(f"ViT output shape: {output_features.shape}")

    # Check number of parameters
    num_params = sum(p.numel() for p in vit_encoder.parameters() if p.requires_grad)
    print(f"ViT number of parameters: {num_params / 1e6:.2f} M") # ViT-Base is ~86M params. This should be similar.

    # Example of a smaller ViT (e.g., ViT-Small)
    vit_s_embed_dim = 384
    vit_s_depth = 12
    vit_s_num_heads = 6
    vit_encoder_small = VisionTransformer(
        img_size=vit_img_size, patch_size=vit_patch_size, in_channels=vit_in_channels,
        embed_dim=vit_s_embed_dim, depth=vit_s_depth, num_heads=vit_s_num_heads,
        mlp_ratio=vit_mlp_ratio, dropout=vit_dropout, num_classes=0
    )
    output_features_small = vit_encoder_small(dummy_cnn_feature_map)
    print(f"Small ViT output shape: {output_features_small.shape}") # (1, 64, 384)
    num_params_small = sum(p.numel() for p in vit_encoder_small.parameters() if p.requires_grad)
    print(f"Small ViT number of parameters: {num_params_small / 1e6:.2f} M") # ViT-Small ~22M params.

    # The PatchEmbedding with patch_size=1 and Conv2d(kernel_size=1) is equivalent to a linear projection
    # of the input channels to the embedding dimension, applied pixel-wise, then flattened.
    # If input is B, C, H, W. proj -> B, E, H, W. flatten -> B, E, H*W. transpose -> B, H*W, E.
    # This matches the TransUNet paper's description of tokenizing the 2D feature map.
    # "The sequence of input patches is obtained by flattening the spatial dimensions of x_pt ∈ R^(C×H'×W')
    #  and then mapping to ViT’s hidden dimension D through a linear projection." (TransUNet paper)
    # Here, C is vit_in_channels, H' and W' are from vit_img_size. D is vit_embed_dim.
    # My PatchEmbedding with Conv2d(kernel_size=1) performs this.

    # Positional embeddings are crucial. They are added to the patch embeddings.
    # The shape is (1, num_patches, embed_dim) and will be broadcasted across the batch.
    # For an 8x8 feature map, num_patches = 64.
    assert vit_encoder.pos_embed.shape == (1, 64, vit_embed_dim)
    assert vit_encoder_small.pos_embed.shape == (1, 64, vit_s_embed_dim)
