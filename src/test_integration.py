import asyncio
import logging
import os
from typing import Dict

import pytest
from google.protobuf.struct_pb2 import Struct
from numpy import float32
from viam.media.video import CameraMimeType
from viam.proto.app.robot import ServiceConfig
from viam.services.vision import Vision

from src.config.config import TrackerConfig
from src.test.fake_camera import FakeCamera
from src.test.fake_detector_vision_service import FakeDetectorVisionService
from src.tracker.track import Track
from src.tracker.tracker import Tracker
from src.tracker_service import TrackerService
from src.utils import decode_image

CAMERA_NAME = "fake-camera"
DETECTOR_NAME = "fake-detector"

PASSING_PROPERTIES = Vision.Properties(
    classifications_supported=True,
    detections_supported=True,
    object_point_clouds_supported=False,
)

MIN_CONFIDENCE_PASSING = 0.8

WORKING_CONFIG_DICT = {
    "camera_name": CAMERA_NAME,
    "detector_name": DETECTOR_NAME,
    "_start_background_loop": False,
}


IMG_PATH = "./src/test/alex"


def get_config(config_dict: Dict) -> ServiceConfig:
    """returns a config populated with picture_directory and camera_name
    attributes.X

    Returns:``
        ServiceConfig: _description_
    """
    struct = Struct()
    struct.update(dictionary=config_dict)
    config = ServiceConfig(attributes=struct)
    return config


def get_vision_service(config_dict: Dict, reconfigure=True):
    service = TrackerService("test")
    cam = FakeCamera(CAMERA_NAME, img_path=IMG_PATH, use_ring_buffer=True)
    camera_name = cam.get_resource_name(CAMERA_NAME)
    detector = FakeDetectorVisionService(DETECTOR_NAME)
    detector_name = detector.get_resource_name(DETECTOR_NAME)
    cfg = get_config(config_dict)
    service.validate_config(cfg)
    if reconfigure:
        service.reconfigure(
            cfg, dependencies={camera_name: cam, detector_name: detector}
        )
    return service


class TestFaceReId:
    @pytest.mark.asyncio
    async def test_person_reid(self):
        service = get_vision_service(WORKING_CONFIG_DICT, reconfigure=True)
        detections = await service.tracker.detector.detect(None)
        assert len(detections) == 2
        assert detections[0].category == "person"
        assert detections[1].category == "car"
        print(detections)
        await service.close()

        # test if len(tracks) ==1
        # track = tracks[0] test that track.re-id-label is alex


if __name__ == "__main__":
    asyncio.run(TestFaceReId().test_person_reid())
