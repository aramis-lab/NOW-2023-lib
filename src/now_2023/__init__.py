# read version from installed package
from importlib.metadata import version
__version__ = version("now_2023")

# populate package namespace
from now_2023.models import ToyModel

__all__ = [
    "ToyModel",
]
