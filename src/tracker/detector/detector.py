from abc import ABC, abstractmethod
from typing import List

from src.config.config import DetectorConfig
from src.image.image import ImageObject
from src.tracker.detector.detection import Detection


class Detector(ABC):
    def __init__(self, cfg: DetectorConfig):
        """
        Initializes the detector with a DetectorConfig object that contains configuration for the model.
        """
        self.cfg = cfg

    @abstractmethod
    async def detect(
        self, image: ImageObject, visualize: bool = False
    ) -> List[Detection]:
        """
        Abstract method to be implemented by specific detector classes. Each detector must implement
        this method to detect objects in the provided image.
        """
