from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from src.config.config import EmbedderConfig
from src.image.image import ImageObject
from src.tracker.detector.detection import Detection
from src.tracker.utils import (
    pad_image_to_target_size,
    resize_for_padding,
)


class Embedder(ABC):
    def __init__(self, cfg: EmbedderConfig):
        """
        Initializes the detector with a DetectorConfig object that contains configuration for the model.
        """
        self.cfg = cfg
        self.distance = cfg.embedder_distance
        self.input_height = cfg.input_height
        self.input_width = cfg.input_width
        self.input_shape = (cfg.input_height, cfg.input_width)

    @abstractmethod
    async def compute_features(
        self, image: ImageObject, detections: List[Detection]
    ) -> List[np.ndarray]:
        """
        Abstract method to be implemented by specific detector classes. Each detector must implement
        this method to detect objects in the provided image.
        """
        pass

    def compute_distance(
        self, feature_vector_1: torch.Tensor, feature_vector_2: torch.Tensor
    ):
        """
        Compute pairwise distances between feature vectors using PyTorch.

        :param feature_vector_1: First feature vector (PyTorch tensor).
        :param feature_vector_2: Second feature vector (PyTorch tensor).
        :return: Computed distance.
        """
        if self.distance == "euclidean":
            distance = torch.norm(feature_vector_1 - feature_vector_2, p=2)
        elif self.distance == "cosine":
            distance = 1 - torch.nn.functional.cosine_similarity(
                feature_vector_1.unsqueeze(0), feature_vector_2.unsqueeze(0)
            )
        elif self.distance == "manhattan":
            distance = torch.sum(torch.abs(feature_vector_1 - feature_vector_2))
        else:
            raise ValueError(f"Unsupported metric '{self.distance}'")
        return distance.cpu().item()

    def crop_detections(
        self, image: ImageObject, detections: List[Detection]
    ) -> torch.Tensor:
        """
        Crop the detections from the image and return a tensor of the cropped images in one batch.
        """
        device = image.float32_tensor.device
        image_height, image_width = image.float32_tensor.shape[
            1:
        ]  # Assuming CxHxW format

        # Stack all bounding boxes into a tensor (x1, y1, x2, y2)
        bboxes = torch.tensor(
            [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]] for d in detections],
            device=device,
        )

        # Crop and resize images
        cropped_images = []

        # TODO: Make this more efficient
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)  # Ensure integer coordinates
            x1, y1 = max(0, x1), max(0, y1)  # Clip to image dimensions
            x2, y2 = min(image_width, x2), min(image_height, y2)

            cropped_image = image.float32_tensor[
                :, y1:y2, x1:x2
            ]  # Crop image (CxH_cropxW_crop)

            # Ensure the cropped region is valid
            if cropped_image.numel() == 0:
                raise ValueError(f"Invalid crop region: {bbox}")

            # Resize the cropped image

            resized_image, new_height, new_width, _, _ = resize_for_padding(
                cropped_image, self.input_shape
            )
            padded_image = pad_image_to_target_size(resized_image, self.input_shape)
            cropped_images.append(padded_image[0])

        # Stack all resized images into a batch
        return torch.stack(cropped_images, dim=0)
