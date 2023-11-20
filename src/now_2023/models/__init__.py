"""The model module.

This module contains the machine learning and deep learning models used in the tutorial.
"""

from ._toy import ToyModel
from ._cnn import CNNModel

__all__ = ["CNNModel", "ToyModel"]
