from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.config.config import DetectorConfig
from src.tracker.detector.detection import Detection


class Detector(ABC):
    def __init__(self, cfg: DetectorConfig):
        """
        Initializes the detector with a DetectorConfig object that contains configuration for the model.
        """
        self.cfg = cfg

    @abstractmethod
    def detect(self, image: np.ndarray, visualize: bool = False) -> List[Detection]:
        """
        Abstract method to be implemented by specific detector classes. Each detector must implement
        this method to detect objects in the provided image.
        """
