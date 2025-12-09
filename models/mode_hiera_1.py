import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEModel, TimesformerModel

class VideoMAE(nn.Module):
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
        self.model = VideoMAEModel.from_pretrained(model_name)
        # self.model = TimesformerModel.from_pretrained(model_name)
        patch_size = self.model.config.patch_size
        tubelet_size = self.model.config.tubelet_size
        old_image_size = self.model.config.image_size
        
        # Update config for new image size
        self.model.config.image_size = new_image_size
        self.model.embeddings.patch_embeddings.image_size = (new_image_size, new_image_size)
        
        # Interpolate position embeddings using video-specific method
        old_pos_embed = self.model.embeddings.position_embeddings.data
        new_pos_embed = self._interpolate_video_embeddings(
            old_image_size, new_image_size, patch_size, tubelet_size, 
            16, old_pos_embed, 'bicubic'
        )
        
        # Concatenate CLS and patches
        # new_pos_embed = torch.cat([cls_pos_embed, new_patch_pos_embed], dim=1)
        self.model.embeddings.position_embeddings = nn.Parameter(new_pos_embed)
        
        # Replace classifier
        hidden_size = self.model.config.hidden_size
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def _interpolate_video_embeddings(
        self, old_image_size, new_image_size, patch_size, tubelet_size,
        num_frames, pos_embedding, interpolation_mode
    ):
        """
        Interpolate position embeddings for video by separating temporal and spatial dimensions.
        """
        n, seq_length, hidden_dim = pos_embedding.shape
        
        # Calculate spatial patches per frame
        old_spatial_patches = (old_image_size // patch_size) ** 2
        
        # Calculate temporal patches
        old_t_patches = seq_length // old_spatial_patches
        
        # Use only complete frames
        usable_patches = old_t_patches * old_spatial_patches
        pos_embedding = pos_embedding[:, :usable_patches, :]
        
        # Reshape to separate temporal frames: (1, T*H*W, D) -> (T, H*W, D)
        pos_embedding = pos_embedding.reshape(old_t_patches, old_spatial_patches, hidden_dim)
        
        # Apply spatial interpolation to each temporal frame
        interpolated_frames = []
        for t in range(old_t_patches):
            # Get one frame: (H*W, D) -> (1, H*W, D)
            frame_embed = pos_embedding[t:t+1, :, :]
            
            # Apply the spatial interpolation function
            interpolated_frame = self._interpolate_embeddings_spatial(
                new_image_size, patch_size, frame_embed, interpolation_mode
            )
            
            interpolated_frames.append(interpolated_frame)
        
        # Stack frames back: T x (1, H*W, D) -> (1, T*H*W, D)
        return torch.cat(interpolated_frames, dim=1)
    
    def _interpolate_embeddings_spatial(
        self,
        image_size: int,
        patch_size: int,
        pos_embedding: torch.Tensor,
        interpolation_mode: str = "bicubic"
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings for pretrained models.
        Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
        """
        n, seq_length, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

        new_seq_length = (image_size // patch_size) ** 2

        if new_seq_length != seq_length:
            pos_embedding = pos_embedding.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))
            if seq_length_1d * seq_length_1d != seq_length:
                raise ValueError(
                    f"seq_length is not a perfect square! Instead got "
                    f"seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d} "
                    f"and seq_length = {seq_length}"
                )

            pos_embedding = pos_embedding.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
            new_seq_length_1d = image_size // patch_size

            new_pos_embedding = nn.functional.interpolate(
                pos_embedding,
                size=new_seq_length_1d,
                mode=interpolation_mode,
                align_corners=True
            )

            new_pos_embedding = new_pos_embedding.reshape(1, hidden_dim, new_seq_length)
            new_pos_embedding = new_pos_embedding.permute(0, 2, 1)

            return new_pos_embedding
        
        return pos_embedding
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.last_hidden_state.mean(dim=1)
        x = self.norm(x)
        x = self.classifier(x)
        return x
    
# if __name__ == "__main__":

#     # Example usage:
#     # python -m hiera-luna25-finetuning.models.model_hiera

#     IMG_SIZE, IMG_DEPTH, MODE = 64, 16, "3D"
#     print(f"{MODE} Hiera:")
#     model = XCLIP(
#         model_name="microsoft/xclip-base-patch32",
#         new_image_size=64,
#         num_classes=1,
#     )
#     print(model)
#     print(f"Output shape: {model(torch.randn(1, 3, IMG_DEPTH, IMG_SIZE, IMG_SIZE)).shape}")
    