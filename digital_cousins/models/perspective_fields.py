import torch
import math
import cv2
from perspective2d import PerspectiveFields as _PerspectiveFields


class PerspectiveFields(torch.nn.Module):
    """
    Thin wrapper around perspective fields model for inferring camera intrinsics
    """
    VERSIONS = {
        "Paramnet-360Cities-edina-centered",
        "Paramnet-360Cities-edina-uncentered",
    }

    def __init__(
            self,
            version="Paramnet-360Cities-edina-uncentered",
            device="cuda",
    ):
        """
        Args:
            version (str): underlying model version to use. Valid options are:
                {"Paramnet-360Cities-edina-uncentered", "Paramnet-360Cities-edina-centered"}
            device (str): device to store tensors on. Default is "cuda"
        """
        # Call super first
        super().__init__()

        # Sanity check version
        assert version in self.VERSIONS,\
            f"Got invalid PerspectiveFields version! Valid options: {self.VERSIONS}, got: {version}"

        # Load model
        self.device = device
        self.model = _PerspectiveFields(version)
        self.model.to(self.device)
        self.model.eval()

    def estimate_camera_intrinsics(self, input_path):
        """
        Estimates the K camera intrinsics matrix using PerspectiveFields model

        Args:
            input_path (str): Absolute path to the image from which to infer camera intrinsics

        Returns:
            np.ndarray: Estimated intrinsics matrix
        """
        # Load the image
        img_bgr = cv2.imread(input_path)

        # Run predictions
        predictions = self.model.inference(img_bgr=img_bgr)

        # Compute intrinsics
        height, width, _ = img_bgr.shape
        # Convert vfov to radians
        vfov_rad = math.radians(predictions['pred_general_vfov'].item())
        # Compute focal length in pixels
        f = (height / 2) / math.tan(vfov_rad / 2)
        # Compute fx and fy
        fx = f * predictions['pred_rel_focal'].item()
        fy = f
        # Compute cx and cy
        cx = width / 2.0
        cy = height / 2.0
        # Construct the camera intrinsics matrix
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        return K
