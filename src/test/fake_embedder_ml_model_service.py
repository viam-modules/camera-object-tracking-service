from typing import Dict, Mapping, Optional

import numpy as np
from viam.services.mlmodel import MLModel
from viam.utils import ValueTypes


class FakeEmbedderMLModel(MLModel):
    """A fake MLModel service that returns mock embeddings for testing purposes."""

    def __init__(self, name: str):
        super().__init__(name)
        # Mock embedding that will be returned
        self._mock_embedding = np.random.rand(512)  # Example 512-dimensional embedding

    async def infer(
        self,
        input_tensors: Dict[str, np.ndarray],
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Return mock embeddings regardless of input.

        Args:
            input_tensors: Dictionary of input tensors (ignored in this fake implementation)
            extra: Optional extra parameters (ignored in this fake implementation)
            timeout: Optional timeout in seconds (ignored in this fake implementation)

        Returns:
            Dictionary containing mock embedding tensor
        """
        return {"embedding": self._mock_embedding}

    async def metadata(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Return metadata about the model.

        Args:
            extra: Optional extra parameters (ignored in this fake implementation)
            timeout: Optional timeout in seconds (ignored in this fake implementation)

        Returns:
            Dictionary containing model metadata
        """
        return {
            "name": "fake_encoder",
            "type": "encoder",
            "inputs": [
                {
                    "name": "image",
                    "description": "Input image tensor",
                    "data_type": "uint8",
                    "shape": [1, 224, 224, 3],  # Example shape
                }
            ],
            "outputs": [
                {
                    "name": "embedding",
                    "description": "Output embedding tensor",
                    "data_type": "float32",
                    "shape": [512],  # Example shape
                }
            ],
        }

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        """Handle arbitrary commands.

        Args:
            command: Dictionary of command parameters
            timeout: Optional timeout in seconds (ignored in this fake implementation)
            **kwargs: Additional keyword arguments (ignored in this fake implementation)

        Returns:
            Dictionary containing command response
        """
        return {"status": "success"}
