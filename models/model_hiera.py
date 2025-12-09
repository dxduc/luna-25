import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification


class VideoMAESmallImageAdapter(nn.Module):
    """
    Adapts a pretrained VideoMAE model to work with smaller input images.
    
    Args:
        model_name: HuggingFace model name or path
        new_image_size: Target image size (e.g., 64 for 64x64 images)
        num_classes: Number of output classes (default: 2 for binary classification)
    """
    
    def __init__(
        self, 
        model_name="MCG-NJU/videomae-large-finetuned-kinetics",
        new_image_size=64,
        num_classes=2
    ):
        super().__init__()
        
        # Load pretrained model
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        
        # Get config values
        patch_size = self.model.config.patch_size
        tubelet_size = self.model.config.tubelet_size
        old_image_size = self.model.config.image_size
        
        # Update config for new image size
        self.model.config.image_size = new_image_size
        self.model.videomae.embeddings.patch_embeddings.image_size = (new_image_size, new_image_size)
        
        # Calculate patch dimensions
        num_frames = 16
        new_h_patches = new_image_size // patch_size
        new_w_patches = new_image_size // patch_size
        new_t_patches = num_frames // tubelet_size
        new_num_patches = new_t_patches * new_h_patches * new_w_patches
        
        old_h_patches = old_image_size // patch_size
        old_w_patches = old_image_size // patch_size
        old_spatial_patches = old_h_patches * old_w_patches
        
        # Interpolate position embeddings
        old_pos_embed = self.model.videomae.embeddings.position_embeddings.data
        embed_dim = old_pos_embed.shape[-1]
        old_num_patches = old_pos_embed.shape[1] - 1  # Exclude CLS
        old_t_patches = old_num_patches // old_spatial_patches
        
        # Extract and reshape patch embeddings
        patch_pos_embed = old_pos_embed[:, 1:, :]
        usable_patches = old_t_patches * old_spatial_patches
        patch_pos_embed = patch_pos_embed[:, :usable_patches, :]
        patch_pos_embed = patch_pos_embed.reshape(1, old_t_patches, old_h_patches, old_w_patches, embed_dim)
        patch_pos_embed = patch_pos_embed.permute(0, 4, 1, 2, 3)
        
        # Interpolate to new size
        new_patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_t_patches, new_h_patches, new_w_patches),
            mode='trilinear',
            align_corners=False
        )
        
        new_patch_pos_embed = new_patch_pos_embed.permute(0, 2, 3, 4, 1)
        new_patch_pos_embed = new_patch_pos_embed.reshape(1, new_num_patches, embed_dim)
        
        # Update position embeddings
        self.model.videomae.embeddings.position_embeddings = nn.Parameter(new_patch_pos_embed)
        
        # Replace classifier
        hidden_size = self.model.config.hidden_size
        self.model.classifier = nn.Sequential(nn.LayerNorm(hidden_size, eps=1e-6),
                                            nn.Linear(hidden_size, num_classes))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            pixel_values: Video tensor of shape (batch_size, num_frames, channels, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        x = x.permute(0, 2, 1, 3, 4)
        outputs = self.model(x)
        return outputs.logits
