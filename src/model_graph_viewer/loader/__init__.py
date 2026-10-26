"""Model loading contract and the OpenVINO IR implementation."""

from .base import ModelLoadError, ModelLoader
from .openvino import OpenVINOLoader

__all__ = ["ModelLoadError", "ModelLoader", "OpenVINOLoader"]
