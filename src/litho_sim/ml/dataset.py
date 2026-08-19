"""
PyTorch dataset for loading wafer / aerial images.

Supports loading clean and defective image directories, with optional
augmentation transforms and a train/val split helper.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

logger = logging.getLogger(__name__)

# Default transform: greyscale, tensor [0, 1]
_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

_AUGMENT_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.05, scale=(0.001, 0.01)),
])


class WaferDataset(Dataset):
    """Load greyscale wafer / aerial images from a folder.

    Parameters
    ----------
    image_folder : str or Path
        Folder containing ``.png`` or ``.tiff`` images.
    transform : callable, optional
        torchvision transform applied to each image.
    extensions : tuple of str
        Accepted file extensions.

    Attributes
    ----------
    image_paths : list of Path
        Sorted list of discovered image paths.
    """

    def __init__(
        self,
        image_folder: str | Path,
        transform: Callable | None = None,
        extensions: tuple[str, ...] = (".png", ".tif", ".tiff"),
    ) -> None:
        self.image_folder = Path(image_folder)
        self.transform = transform or _DEFAULT_TRANSFORM
        self.extensions = extensions

        self.image_paths: list[Path] = sorted(
            p for p in self.image_folder.iterdir()
            if p.suffix.lower() in extensions
        )

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images with extensions {extensions} found in '{image_folder}'."
            )
        logger.info(
            "WaferDataset: %d images in '%s'",
            len(self.image_paths),
            self.image_folder,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        # Label: 0=clean, 1=defect (inferred from filename convention)
        label = 1 if "defect" in path.stem.lower() else 0
        return tensor, label


def create_dataloaders(
    train_folder: str | Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 2,
    augment: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from a single folder.

    Parameters
    ----------
    train_folder : str or Path
        Folder with training images (clean only for autoencoder).
    batch_size : int
        Mini-batch size.
    val_split : float
        Fraction of images reserved for validation (0–1).
    num_workers : int
        DataLoader worker threads.
    augment : bool
        Apply random augmentation to the training split.
    seed : int
        RNG seed for the split.

    Returns
    -------
    train_loader, val_loader : DataLoader, DataLoader
    """
    full_dataset = WaferDataset(
        train_folder,
        transform=_AUGMENT_TRANSFORM if augment else _DEFAULT_TRANSFORM,
    )
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Val split always uses default (non-augmented) transform
    val_ds.dataset = WaferDataset(train_folder, transform=_DEFAULT_TRANSFORM)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(
        "DataLoaders: train=%d, val=%d (batch_size=%d)",
        len(train_ds), len(val_ds), batch_size,
    )
    return train_loader, val_loader

