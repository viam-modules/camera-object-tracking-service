import asyncio
import logging
import os
from typing import Dict

import pytest
import pytest_asyncio
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ServiceConfig
from viam.services.vision import Vision

from src.image.image import ImageObject
from src.test.fake_camera import FakeCamera
from src.test.fake_detector_vision_service import FakeDetectorVisionService
from src.test.fake_embedder_ml_model_service import FakeEmbedderMLModel
from src.tracker_service import TrackerService

CAMERA_NAME = "fake-camera"
DETECTOR_NAME = "fake-detector"
EMBEDDER_NAME = "fake-embedder"

PASSING_PROPERTIES = Vision.Properties(
    classifications_supported=True,
    detections_supported=True,
    object_point_clouds_supported=False,
)

MIN_CONFIDENCE_PASSING = 0.8

WORKING_CONFIG_DICT = {
    "camera_name": CAMERA_NAME,
    "detector_name": DETECTOR_NAME,
    "embedder_name": EMBEDDER_NAME,
    "_start_background_loop": False,
    "embedder_output_name": "output",
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

    embedder = FakeEmbedderMLModel(EMBEDDER_NAME)
    embedder_name = embedder.get_resource_name(EMBEDDER_NAME)

    cfg = get_config(config_dict)
    service.validate_config(cfg)
    if reconfigure:
        service.reconfigure(
            cfg,
            dependencies={
                camera_name: cam,
                detector_name: detector,
                embedder_name: embedder,
            },
        )
    return service


class TestTracker:
    @pytest_asyncio.fixture(autouse=True)
    async def setup_service(self):
        self.service = get_vision_service(WORKING_CONFIG_DICT, reconfigure=True)
        imgs, _ = await self.service.tracker.camera.get_images()
        if imgs is None or len(imgs) == 0:
            raise ValueError("No images returned by get_images")
        self.img = imgs[0]
        self.image_object = ImageObject(self.img)
        self.service.tracker.last_image = self.image_object
        yield
        # Clean up after tests
        await self.service.close()  # Close any open connections

    @pytest.mark.asyncio
    async def test_detector(self):
        # Test detection from vision service
        image_object = ImageObject(self.img)
        detections = await self.service.tracker.detector.detect(image_object)
        assert len(detections) == 2
        assert detections[0].category == "person"
        assert detections[1].category == "car"

    @pytest.mark.asyncio
    async def test_embedder(self):
        # Test embeddings from mlmodel service
        detections = await self.service.tracker.detector.detect(self.image_object)
        embeddings = await self.service.tracker.embedder.compute_features(
            self.image_object, detections
        )
        assert len(embeddings) == 2
        assert embeddings[0].shape == (512,)
        assert embeddings[1].shape == (512,)

    @pytest.mark.asyncio
    async def test_tracker(self):
        await self.service.tracker.update(self.image_object)
        dets = self.service.tracker.get_current_detections()
        assert len(dets) == 2

    @pytest.mark.asyncio
    async def test_get_detections_from_camera(self):
        for i in range(self.service.tracker.minimum_track_persistance + 2):
            await self.service.tracker.update(self.image_object)
        dets = await self.service.get_detections_from_camera(
            camera_name=CAMERA_NAME, timeout=0, extra=None
        )
        assert len(dets) == 2
        assert dets[0].class_name.startswith("person")
        assert dets[1].class_name.startswith("car")

    @pytest.mark.asyncio
    async def test_get_detections_from_camer_with_empty_camera_name(self):
        for i in range(self.service.tracker.minimum_track_persistance + 2):
            await self.service.tracker.update(self.image_object)
        dets = await self.service.get_detections_from_camera(
            camera_name="", timeout=0, extra=None
        )
        assert len(dets) == 2
        assert dets[0].class_name.startswith("person")
        assert dets[1].class_name.startswith("car")


if __name__ == "__main__":
    # Run all tests with pytest
    pytest.main(
        ["-xvs", __file__]
    )  # verbose, stop after first failure, don't capture output
