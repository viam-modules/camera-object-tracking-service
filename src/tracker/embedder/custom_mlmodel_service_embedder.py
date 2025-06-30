from typing import List

import numpy as np
import torch
from viam.logging import getLogger
from viam.services.mlmodel import MLModel

from src.config.config import EmbedderConfig
from src.image.image import ImageObject
from src.tracker.detector.detection import Detection
from src.tracker.embedder.embedder import Embedder

LOGGER = getLogger(__name__)


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
        for img_np in cropped_images:
            try:
                result = await self.ml_model.infer({self.input_name: img_np})
            except Exception as e:
                LOGGER.error(
                    f"Error getting embedding from MLModel service {self.ml_model.name}: {e}"
                )
                return []
            embedding = result[self.output_name]
            features.append(torch.from_numpy(embedding))

        return features
