import random
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int
    z: int


class BaseCrop:
    def __init__(self, random_shift: int = 0):
        self.random_shift = random_shift
        self.train_mode = True
        self.bottom_left_point: Optional[Point] = None
        self.upper_right_point: Optional[Point] = None

    def _get_shift(self) -> int:
        """Return a random shift in train mode and no shift otherwise."""
        return (
            random.randint(-self.random_shift, self.random_shift)
            if self.train_mode
            else 0
        )

    def _get_slice(self, left_offset: int, right_offset: int):
        """Return a slice in a single dimension with possibly a random shift."""
        shift = self._get_shift()
        return slice(left_offset + shift, right_offset + shift)

    def _get_bbox(self) -> tuple:
        """Return the bounding box to use on the input image."""
        if self.bottom_left_point is None or self.upper_right_point is None:
            raise ValueError("No bounding box configured")
        return (
            Ellipsis,
            self._get_slice(self.bottom_left_point.x, self.upper_right_point.x),
            self._get_slice(self.bottom_left_point.y, self.upper_right_point.y),
            self._get_slice(self.bottom_left_point.z, self.upper_right_point.z),
        )

    def __call__(self, img):
        return img[self._get_bbox()].clone()

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


class CropLeftHC(BaseCrop):
    """Crops the left hippocampus of a MRI non-linearly registered to MNI"""

    def __init__(self, random_shift: int = 0):
        super().__init__(random_shift=random_shift)
        self.bottom_left_point = Point(25, 50, 27)
        self.upper_right_point = Point(55, 90, 57)


class CropRightHC(BaseCrop):
    """Crops the right hippocampus of a MRI non-linearly registered to MNI"""

    def __init__(self, random_shift: int = 0):
        super().__init__(random_shift=random_shift)
        self.bottom_left_point = Point(65, 50, 27)
        self.upper_right_point = Point(95, 90, 57)
