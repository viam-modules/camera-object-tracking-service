from typing import List

import numpy as np
import torch
from viam.services.mlmodel import MLModel

from src.config.config import EmbedderConfig
from src.image.image import ImageObject
from src.tracker.detector.detection import Detection
from src.tracker.embedder.embedder import Embedder


class CustomMLModelServiceEmbedder(Embedder):
    def __init__(self, cfg: EmbedderConfig, ml_model: MLModel):
        """Initialize the embedder with a Viam MLModel Service.

        Args:
            cfg: Embedder configuration
            ml_model: A Viam MLModel Service instance that implements infer()
        """
        super().__init__(cfg)
        self.ml_model = ml_model
        self.input_name = cfg.input_name
        self.output_name = cfg.output_name

    async def compute_features(
        self, image: ImageObject, detections: List[Detection]
    ) -> List[torch.Tensor]:
        """Compute feature embeddings for each detection using the MLModel service.

        Args:
            image: The input image as a numpy array
            detections: List of Detection objects to compute features for

        Returns:
            List of feature embeddings as numpy arrays
        """
        # Use the parent class's crop_detections method to crop the detections on
        cropped_images = self.crop_detections(image, detections)

        features = []
        for cropped_image in cropped_images:
            # Convert PyTorch tensor to numpy array and prepare for MLModel
            # TODO: define a convention for input shape
            img_np = cropped_image.numpy()

            # Get embedding from MLModel
            result = await self.ml_model.infer({self.input_name: img_np})
            embedding = result[self.output_name]

            features.append(torch.from_numpy(embedding))

        return features
