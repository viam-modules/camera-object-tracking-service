from typing import List, Optional

from viam.media.video import ViamImage
from viam.services.vision import Vision
from viam.utils import ValueTypes

from src.config.config import DetectorConfig
from src.image.image import ImageObject
from src.tracker.detector.detection import Detection
from src.tracker.detector.detector import Detector


class CustomVisionServiceDetector(Detector):
    def __init__(self, cfg: DetectorConfig, vision_service: Vision):
        """Initialize the detector with a Viam Vision Service.

        Args:
            cfg: Detector configuration
            vision_service: A Viam Vision Service instance that implements get_detections()
        """
        super().__init__(cfg)
        self.vision_service = vision_service
        self.chosen_labels = cfg.chosen_labels

    async def detect(
        self, image: ImageObject, visualize: bool = False
    ) -> List[Detection]:
        """Detect objects in the image using the Viam Vision Service.

        Args:
            image: The input image
            visualize: Whether to visualize the detections (ignored in this implementation)

        Returns:
            List of Detection objects
        """
        # Get detections from the vision service
        viam_detections = await self.vision_service.get_detections(None)

        # Convert Viam detections to our custom Detection type
        detections = []
        for viam_det in viam_detections:
            # Check if we have a threshold for this class and if confidence exceeds it
            if self.chosen_labels is None or (
                viam_det.class_name in self.chosen_labels
                and viam_det.confidence > self.chosen_labels[viam_det.class_name]
            ):
                bbox = [
                    int(viam_det.x_min),
                    int(viam_det.y_min),
                    int(viam_det.x_max),
                    int(viam_det.y_max),
                ]
                detections.append(
                    Detection(
                        bbox=bbox,
                        score=viam_det.confidence,
                        category=viam_det.class_name,
                    )
                )

        return detections
