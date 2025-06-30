from typing import List, Mapping, Optional

from viam.media.video import ViamImage
from viam.proto.service.vision import Classification, Detection
from viam.services.vision import Vision
from viam.utils import ValueTypes


class FakeDetectorVisionService(Vision):
    """A fake vision service that returns mock detections for testing purposes."""

    def __init__(self, name: str):
        super().__init__(name)
        # Mock detections that will be returned
        self._mock_detections = [
            Detection(
                x_min=100,
                y_min=100,
                x_max=200,
                y_max=200,
                confidence=0.95,
                class_name="person",
            ),
            Detection(
                x_min=300,
                y_min=150,
                x_max=400,
                y_max=250,
                confidence=0.85,
                class_name="car",
            ),
        ]

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        """Return mock detections regardless of input image.

        Args:
            image: The input image (ignored in this fake implementation)
            extra: Optional extra parameters (ignored in this fake implementation)
            timeout: Optional timeout in seconds (ignored in this fake implementation)

        Returns:
            List of mock Detection objects
        """
        return self._mock_detections

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        """Return mock detections for camera input.

        Args:
            camera_name: The name of the camera (ignored in this fake implementation)
            extra: Optional extra parameters (ignored in this fake implementation)
            timeout: Optional timeout in seconds (ignored in this fake implementation)

        Returns:
            List of mock Detection objects
        """
        return self._mock_detections

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        """Return empty classifications list.

        Args:
            image: The input image (ignored in this fake implementation)
            count: Number of classifications to return (ignored in this fake implementation)
            extra: Optional extra parameters (ignored in this fake implementation)
            timeout: Optional timeout in seconds (ignored in this fake implementation)

        Returns:
            Empty list of classifications
        """
        return NotImplementedError

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        """Return empty classifications list for camera input.

        Args:
            camera_name: The name of the camera (ignored in this fake implementation)
            count: Number of classifications to return (ignored in this fake implementation)
            extra: Optional extra parameters (ignored in this fake implementation)
            timeout: Optional timeout in seconds (ignored in this fake implementation)

        Returns:
            Empty list of classifications
        """
        return NotImplementedError

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List:
        """Return empty point clouds list.

        Args:
            camera_name: The name of the camera (ignored in this fake implementation)
            extra: Optional extra parameters (ignored in this fake implementation)
            timeout: Optional timeout in seconds (ignored in this fake implementation)

        Returns:
            Empty list of point clouds
        """
        return NotImplementedError

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Return properties indicating which features are supported.

        Args:
            extra: Optional extra parameters (ignored in this fake implementation)
            timeout: Optional timeout in seconds (ignored in this fake implementation)

        Returns:
            Dictionary indicating supported features
        """
        return {
            "classifications_supported": False,
            "detections_supported": True,
            "object_point_clouds_supported": False,
        }

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ):
        """Return a CaptureAllResult with requested data.

        Args:
            camera_name: The name of the camera (ignored in this fake implementation)
            return_image: Whether to return the image (ignored in this fake implementation)
            return_classifications: Whether to return classifications (ignored in this fake implementation)
            return_detections: Whether to return detections
            return_object_point_clouds: Whether to return point clouds (ignored in this fake implementation)
            extra: Optional extra parameters (ignored in this fake implementation)
            timeout: Optional timeout in seconds (ignored in this fake implementation)

        Returns:
            CaptureAllResult with requested data
        """

        return NotImplementedError
        from viam.services.vision.vision import CaptureAllResult

        return CaptureAllResult(
            image=None,
            classifications=None,
            detections=self._mock_detections if return_detections else None,
            objects=None,
            extra=None,
        )
