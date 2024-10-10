import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch

import digital_cousins
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as _DepthAnythingV2

from digital_cousins.utils.processing_utils import process_depth_linear, unprocess_depth_linear


class DepthAnythingV2(torch.nn.Module):
    """
    Thin wrapper around DepthAnything V2 model for inferring depth maps
    """
    # Maps backbone size to configuration to use
    CONFIGS = {
        'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # Maps dataset name to max depth to use
    DATASET_MAX_DEPTHS = {
        "hypersim": 20,
        "vkitti": 80,
    }

    def __init__(
            self,
            backbone_size="large",
            backbone_dataset="hypersim",
            max_depth=None,
            device="cuda",
    ):
        """
        Args:
            backbone_size (str): Size of underlying ViT model to use. Options are {'small', 'base', 'large'}
            backbone_dataset (str): Which training dataset the underlying model was trained on.
                Use "hypersim" for indoor estimation, "vkitti" for outdoor model
            max_depth (None or int): Maximum depth (m) to use for depth estimation. If None, will use a default
                based on @backbone_dataset
            device (str): device to store tensors on. Default is "cuda"
        """
        # Call super first
        super().__init__()

        # Sanity check values
        assert backbone_size in self.CONFIGS,\
            f"Got invalid DepthAnythingV2 backbone_size! Valid options: {self.CONFIGS.keys()}, got: {backbone_size}"
        assert backbone_dataset in self.DATASET_MAX_DEPTHS,\
            f"Got invalid DepthAnythingV2 backbone_dataset! Valid options: {self.DATASET_MAX_DEPTHS.keys()}, got: {backbone_dataset}"

        self.max_depth = self.DATASET_MAX_DEPTHS[backbone_dataset] if max_depth is None else max_depth
        self.device = device

        # Load model
        self.model = _DepthAnythingV2(**{**self.CONFIGS[backbone_size], "max_depth": self.max_depth})
        self.model.load_state_dict(
            torch.load(
                f"{digital_cousins.CHECKPOINT_DIR}/depth_anything_v2_metric_{backbone_dataset}_{self.CONFIGS[backbone_size]['encoder']}.pth",
                map_location=self.device,
                weights_only=True,
            ),
        )
        self.model.to(self.device).eval()

    def estimate_depth_linear(self, input_path, output_path, depth_limits=(0, 10.0)):
        """
        Estimates linear depth map using DepthAnythingV2 model.

        NOTE: depth linear implies that the outputted map is taken with respect to the image plane,
            NOT the camera position!

        Args:
            input_path (str): Absolute path to the image from which to infer linear depth map
            output_path (str): Absolute path to the location for the estimated linear depth map image
            depth_limits (2-tuple): (min, max) values from the depth map to normalize scaling when saving the
                depth map image

        Returns:
            np.ndarray: Estimated linear depth map
        """
        img = cv2.imread(input_path)
        H, W, _ = img.shape
        pred = self.model.infer_image(img)  # HxW depth map in meters in numpy

        # Save output; normalize and convert the depth image to an 8-bit format if necessary
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        depth = process_depth_linear(depth=pred, in_limits=depth_limits)
        Image.fromarray(depth).save(output_path)

        return depth

    def crop_center(self, image, crop_percent=1):
        """
        Crops the center of the image based on the specified crop percentage.
        Works for both single-channel (grayscale/depth) and three-channel (RGB) images.

        Params:
            image (np.ndarray): A numpy array representing the image. Can be 2D (single-channel) or 3D (multi-channel).
            crop_percent (int): The percentage of the image to retain in the center. Default is 0.8.

        Returns:
        - np.ndarray: A numpy array representing the cropped image.
        """

        if crop_percent <= 0 or crop_percent > 1:
            raise ValueError("crop_percent must be between 0 and 1")

        # Check the number of dimensions to handle both single and multi-channel images
        if image.ndim == 3:  # Multi-channel image
            height, width, _ = image.shape
        elif image.ndim == 2:  # Single-channel image
            height, width = image.shape
        else:
            raise ValueError("Unsupported image shape")

        # Calculate the new dimensions
        new_height = int(height * crop_percent)
        new_width = int(width * crop_percent)

        # Calculate margins
        top_margin = (height - new_height) // 2
        left_margin = (width - new_width) // 2

        # Crop the image based on the number of dimensions
        if image.ndim == 3:
            cropped_image = image[top_margin:top_margin + new_height, left_margin:left_margin + new_width, :]
        else:  # image.ndim == 2
            cropped_image = image[top_margin:top_margin + new_height, left_margin:left_margin + new_width]

        print(f"cropped_img_shape: {cropped_image.shape}")
        return cropped_image
