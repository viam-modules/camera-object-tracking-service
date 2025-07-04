import io
from typing import Optional

import numpy as np
import torch
from PIL import Image
from viam.media.video import ViamImage


def get_tensor_from_np_array(np_array: np.ndarray) -> torch.Tensor:
    """
    returns an RGB tensor

    If crop_region is provided, it should contain relative coordinates (0.0-1.0)
    for x1, y1, x2, y2 that will be converted to absolute pixel positions
    """
    uint8_tensor = (
        torch.from_numpy(np_array).permute(2, 0, 1).contiguous()
    )  # -> to (C, H, W)
    float32_tensor = uint8_tensor.to(dtype=torch.float32)
    return uint8_tensor, float32_tensor


class ImageObject:
    def __init__(
        self,
        viam_image: ViamImage,
        pil_image: Optional[Image.Image] = None,
        crop_region: Optional[dict] = None,
    ):
        """
        Initialize an ImageObject from a ViamImage.

        Args:
            viam_image: A ViamImage object containing the image data
            pil_image: Optional PIL Image object. If not provided, will be created from viam_image
            crop_region: Optional dictionary with relative coordinates (0.0-1.0) for cropping.
                        Should contain keys: x1_rel, y1_rel, x2_rel, y2_rel

        Attributes:
            viam_image: The original ViamImage object
            pil_image: PIL Image representation in RGB format
            np_array: NumPy array representation of the image (H, W, C) in uint8
            uint8_tensor: PyTorch tensor in uint8 format (C, H, W) on specified device
            float32_tensor: PyTorch tensor in float32 format (C, H, W) on specified device
            width: Width of the image in pixels
            height: Height of the image in pixels
            device: The device (CPU/CUDA) where tensors are stored
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pil_image is not None:
            self.pil_image = pil_image
        if viam_image is not None:
            self.viam_image = viam_image
            self.pil_image = Image.open(io.BytesIO(viam_image.data)).convert(
                "RGB"
            )  # -> in (H, W, C)

        self.np_array = np.array(self.pil_image, dtype=np.uint8)
        uint8_tensor, float32_tensor = get_tensor_from_np_array(self.np_array)
        self.uint8_tensor = uint8_tensor.to(self.device)
        self.float32_tensor = float32_tensor.to(self.device)

        self.width = uint8_tensor.shape[2]
        self.height = uint8_tensor.shape[1]

        if crop_region is not None:
            # Convert relative coordinates (0.0-1.0) to absolute pixel positions

            x1 = int(crop_region.get("x1_rel", 0.0) * self.width)
            y1 = int(crop_region.get("y1_rel", 0.0) * self.height)
            x2 = int(crop_region.get("x2_rel", 1.0) * self.width)
            y2 = int(crop_region.get("y2_rel", 1.0) * self.height)

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, self.width - 1))
            y1 = max(0, min(y1, self.height - 1))
            x2 = max(x1 + 1, min(x2, self.width))
            y2 = max(y1 + 1, min(y2, self.height))

            # Apply cropping to both tensors
            self.uint8_tensor = self.uint8_tensor[:, y1:y2, x1:x2]
            self.float32_tensor = self.float32_tensor[:, y1:y2, x1:x2]
