import numpy as np

import torch
import torchvision
from torchvision.transforms import Resize, InterpolationMode, Normalize

from digital_cousins.models.visual_encoder import VisualEncoder


class DinoV2Encoder(VisualEncoder):

    BACKBONE_ARCHES = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }

    EMBEDDING_DIMS = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    def __init__(
            self,
            backbone_size="small",
            aspect_ratio=(4, 3),
            feature_scale=1,
            batch_size=32,
            preprocess_batch_size=None,
            device="cuda",
    ):
        """
        Args:
            backbone_size (str): Size of the backbone model. Valid options are: "small", "base", "large", "giant".
                Default is "small", which we've found to work empirically better than all other models
                (and, conveniently, is also the fastest!)
            aspect_ratio (2-tuple): (W, H) aspect ratio to convert input image into during preprocessing phase
            feature_scale (int): Scaling factor for the outputted encoded feature patches. This will scale the
                dimensions of the outputted feature patch.
            batch_size (None or int): If specified, batch size to use when computing features for a given set of images.
                This is used to avoid OOM errors with limited VRAM
            preprocess_batch_size (None or int): If specified, batch size to use when preprocessing images before
                passing them into the encoder. This is used to avoid OOM errors with limited VRAM
            device (str): device to store tensors on. Default is "cuda"
        """
        # Always run super first
        super().__init__(
            batch_size=batch_size,
            preprocess_batch_size=preprocess_batch_size,
            device=device,
        )

        # Store additional
        self.aspect_ratio = aspect_ratio
        self.feature_scale = int(feature_scale)

        # Sanity check backbone size
        assert backbone_size in self.BACKBONE_ARCHES, \
            f"Got invalid dinov2 backbone size: {backbone_size}. Valid options are: {self.BACKBONE_ARCHES.keys()}"
        backbone_name = f"dinov2_{self.BACKBONE_ARCHES[backbone_size]}"
        self.backbone_size = backbone_size

        # Load the encoder backbone
        self.backbone = torch.hub.load(repo_or_dir='facebookresearch/dinov2', model=backbone_name)
        self.backbone.eval()
        self.backbone.to(self.device)

    @property
    def feature_width(self):
        """
        Returns:
            int: Width of outputted feature maps
        """
        return 14 * self.aspect_ratio[0] * self.feature_scale

    @property
    def feature_height(self):
        """
        Returns:
            int: Height of outputted feature maps
        """
        return 14 * self.aspect_ratio[1] * self.feature_scale

    @property
    def embedding_dim(self):
        return self.EMBEDDING_DIMS[self.backbone_size]

    def preprocess(self, x):
        # Standardize shape to be 4-dim (B, H, W, C)
        if len(x.shape) < 4:
            x = x.reshape(1, *x.shape)

        # Convert from range [0, 255] -> [0.0, 1.0]
        x = torch.tensor(x, requires_grad=False) / 255.0

        # Permute, resize, normalize
        new_width = self.feature_width * 14
        new_height = self.feature_height * 14
        return torchvision.transforms.Compose(
            [
                torchvision.ops.Permute([0, 3, 1, 2]),
                Resize((new_height, new_width), interpolation=InterpolationMode.BICUBIC),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(x)

    def forward(self, x):
        _, _, H, W = x.shape
        return self.backbone.forward_features(x)["x_norm_patchtokens"].view(-1, H // 14, W // 14, self.embedding_dim)
