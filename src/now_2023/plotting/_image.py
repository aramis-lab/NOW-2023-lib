import torch
from pathlib import Path
from typing import Optional


def _show_slices(slices):
    """Function to display a row of image slices"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def plot_image(
    image: Path, cut_coords: tuple[int, int, int], title: Optional[str] = None
):
    """Plot a slice of a brain image.

    Paramaters
    ----------
    image : Path
        The path to the nifti image to be plotted.

    cut_coords : :obj:`tuple` of three :obj:`int`
        The coordinates of the slice to be displayed.

    title : str, optional
        The title of the plot.
    """
    import nibabel as nib
    import matplotlib.pyplot as plt
    from scipy.ndimage import rotate

    data = nib.load(image).get_fdata()
    slice_0 = data[:, :, cut_coords[0]]
    slice_1 = data[cut_coords[1], :, :]
    slice_2 = data[:, cut_coords[2], :]

    _show_slices([slice_0, rotate(slice_1, 90), rotate(slice_2, 90)])

    if title:
        plt.suptitle(title)

    plt.show()


def plot_tensor(
    tensor: Path | torch.Tensor,
    cut_coords: tuple[int, int, int],
    title: Optional[str] = None,
):
    """Plot a tensor image.

    Parameters
    ----------
    tensor : Path ot torch.Tensor
        The tensor image to be displayed. This can be a path to a ".pt" tensor file,
        or a torch.Tensor object.

    cut_coords : tuple of three ints
        The coordinates of the slice to be displayed.

    title : str, optional
        The title of the plot.
    """
    import torch
    import matplotlib.pyplot as plt

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.load(tensor)
    slice_0 = tensor[0, cut_coords[0], :, :]
    slice_1 = tensor[0, :, cut_coords[1], :]
    slice_2 = tensor[0, :, :, cut_coords[2]]

    _show_slices([slice_0, slice_1, slice_2])

    if title:
        plt.suptitle(title)

    plt.show()


def plot_hc(dataset_folder: Path, subject: str, hemi: str, cut_coords: tuple = None):
    """Plot a brain image of the Hipocampus.

    This is a high-level function which assumes some structure in the input data.

    Parameters
    ----------
    dataset_folder : Path
        The path to the folder in which the raw input dataset of the tutorial was downloaded.

    subject : str
        The identifier of the subject for which a plot is desired.

    hemi : str
        Either "left" or "right". This indicates which hipocampus should be displayed.

    cut_coords : tuple of three ints
        The coordinates of the slice to be displayed.
    """
    image_folder = (
        dataset_folder
        / "subjects"
        / subject
        / "ses-M00"
        / "deeplearning_prepare_data"
        / "image_based"
        / "custom"
    )
    image_filename = f"{subject}_ses-M00_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability_{hemi}.pt"
    tensor = torch.load(image_folder / image_filename)
    plot_tensor(
        tensor,
        cut_coords=cut_coords or (15, 20, 15),
        title=f"Center slices of {hemi} HC of subject {subject}",
    )
