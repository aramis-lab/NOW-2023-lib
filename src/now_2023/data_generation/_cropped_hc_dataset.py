import torch
from pathlib import Path
from typing import Optional
from now_2023.utils import CropLeftHC, CropRightHC


def generate_cropped_hc_dataset(
    raw_dataset_folder: Path,
    hemi: str,
    output_folder: Optional[Path] = None,
    verbose: bool = True,
) -> None:
    """Generate a version of the cropped HC dataset.

    This function works with the raw dataset and will crop the available images.

    Parameters
    ----------
    raw_dataset_folder : Path
        The path to the raw dataset downloaded.

    hemi : str
        Either "left" or "right". Will crop the associated hipocampus.

    output_folder : Path, optional
        If specified, the output dataset will be written in this folder, under
        a "subjects" subfolder.
        Otherwise, the output dataset will be written in the same folder as the
        raw input dataset, under "cropped/subjects".

    verbose : bool, optional
        If True, the function will print information to stdout.
    """
    subjects_folder = raw_dataset_folder / "CAPS" / "subjects"
    if output_folder is None:
        output_folder = raw_dataset_folder / "cropped" / "subjects"
    else:
        output_folder = output_folder / "subjects"
    if verbose:
        print(f"Cropped images will be written in {output_folder}")
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    subjects = [_.name for _ in subjects_folder.iterdir()]

    for subject in subjects:
        if verbose:
            print(f"Generating HC ({hemi}) cropped images for subject {subject}...")
        input_image_folder = (
            subjects_folder
            / subject
            / "ses-M00"
            / "deeplearning_prepare_data"
            / "image_based"
            / "custom"
        )
        input_image_filename = f"{subject}_ses-M00_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.pt"
        try:
            preprocessed_pt = torch.load(input_image_folder / input_image_filename)
        except:
            if verbose:
                print(
                    f"!!! Error reading input tensor for subject {subject}. Skipping..."
                )
            continue

        cropper = CropLeftHC(2) if hemi == "left" else CropRightHC(2)
        hc_cropped = cropper(preprocessed_pt)

        save_path = (
            output_folder
            / subject
            / "ses-M00"
            / "deeplearning_prepare_data"
            / "image_based"
            / "custom"
        )
        if not save_path.exists():
            save_path.mkdir(parents=True)
        if verbose:
            print(f"Saving {hemi} HC cropped image in {save_path}...")
        torch.save(
            hc_cropped,
            save_path
            / f"{subject}_ses-M00_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability_{hemi}.pt",
        )
