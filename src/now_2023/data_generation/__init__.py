"""The data generation module."""

from ._toy_dataset import generate_toy_dataset
from ._cropped_hc_dataset import generate_cropped_hc_dataset

__all__ = ["generate_toy_dataset", "generate_cropped_hc_dataset"]
