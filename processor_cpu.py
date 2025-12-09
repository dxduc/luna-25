import logging
import os

import numpy as np
import torch
from torchvision.transforms import v2

import dataloader
from models.mode_hiera_1 import VideoMAE


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)


class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, mode="3D", suppress_logs=False, model_name="finetune-hiera"):

        self.size_px = 64
        self.size_mm = 50
        self.depth_px = 16

        self.model_name = model_name
        self.mode = mode
        self.suppress_logs = suppress_logs

        # if not self.suppress_logs:
        #     logging.info("Initializing the deep learning system")

        # if self.mode == "3D" and "hiera" in self.model_name.lower():
        #     self.model_3d = Hiera3D(
        #         image_size=self.size_px, image_depth=self.depth_px, kind="finetuned")
        # elif self.mode == "2D" and "hiera" in self.model_name.lower():
        #     self.model_2d = Hiera2D(
        #         image_size=self.size_px, kind="finetuned")
        # elif self.mode == "2D" and "vit" in self.model_name.lower():
        #     self.model_2d = ViT2D(image_size=self.size_px,
        #                           kind="finetuned")
        # else:
        #     raise ValueError("Invalid mode and/or model_name.")

        # self.model_root = "/opt/app/resources/"
        # self.model_3d = VideoMAESmallImageAdapter(
        #     model_name="MCG-NJU/videomae-large-finetuned-kinetics",
        #     new_image_size=64,
        #     num_classes=1
        # )
        
        self.model_3d = VideoMAE(
            model_name="MCG-NJU/videomae-large-finetuned-kinetics",
            new_image_size=64,
            num_classes=1,
        )

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):

        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = dataloader.clip_and_scale(patch)

        # normalize
        normalize_transform = v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.mode == "2D" else \
            v2.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        patch = normalize_transform(torch.from_numpy(patch))
        if self.mode == "3D":
            # transpose to (Channel, Depth, Height, Width)
            patch = patch.permute(1, 0, 2, 3)
        patch = patch.detach().cpu().numpy()

        return patch

    def _process_model(self, mode):

        if not self.suppress_logs:
            logging.info("Processing in %s", mode)

        if mode == "3D":
            output_shape = [self.depth_px, self.size_px, self.size_px]
            model = self.model_3d
        else:
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_2d

        nodules = []

        for _coord in self.coords:

            patch = self.extract_patch(_coord, output_shape, mode=mode)
            nodules.append(patch)

        nodules = np.array(nodules)
        #nodules = torch.from_numpy(nodules).cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nodules = torch.from_numpy(nodules).to(device)

        ckpt_path = r"models/videomae_large.pth"
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt)
        model.eval()
        with torch.no_grad():
            logits = model(nodules)
        logits = logits.data.cpu().numpy()

        logits = np.array(logits)
        return logits

    def predict(self):
        logits = self._process_model(self.mode)
        # logits = self.model_3d.forward()
        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probability, logits
