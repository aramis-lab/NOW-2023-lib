import torch
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, sampler
from typing import Optional, Callable
from dataclasses import dataclass


__all__ = ["Sample", "MRIDataset"]


@dataclass
class Sample:
    image: torch.Tensor
    label: str
    participant_id: str
    session_id: str
    hemi: str


class MRIDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        data_df: pd.DataFrame,
        transform: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        img_dir : str
            Path to the CAPS directory containing preprocessed images.

        data_df : DataFrame
            Metadata of the population.
            Columns include `participant_id`, `session_id`, and `diagnosis`.

        transform : Callable, optional
            List of transforms applied on-the-fly, chained with torchvision.transforms.Compose.
        """
        self.img_dir: Path = Path(img_dir)
        self.transform = transform
        self.data_df = data_df
        self.label_code = {"AD": 1, "CN": 0}
        self.size = self[0]["image"].shape

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> dict:
        return self._get_sample(idx).__dict__

    def _get_sample(self, idx: int) -> Sample:
        diagnosis = self.data_df.loc[idx, "diagnosis"]
        label = self.label_code[diagnosis]
        participant_id = self.data_df.loc[idx, "participant_id"]
        session_id = self.data_df.loc[idx, 'session_id']
        hemi = self.data_df.loc[idx, "hemi"]

        image_filename = f"{participant_id}_{session_id}_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability_{hemi}.pt"
        image_folder = self.img_dir / "subjects" / participant_id / session_id / "deeplearning_prepare_data" / "image_based" / "custom"
        image = torch.load(image_folder / image_filename)

        if self.transform:
            image = self.transform(image)

        return Sample(image, label, participant_id, session_id, hemi)

    def train(self):
        if self.transform:
            self.transform.train()

    def eval(self):
        if self.transform:
            self.transform.eval()


