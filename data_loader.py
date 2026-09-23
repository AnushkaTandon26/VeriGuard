from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class RealFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ["fake", "real"]
        self.samples = []

        for label_idx, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label_idx))

        # Filter unreadable files once to avoid runtime failures during training.
        self.samples = self._filter_valid_samples(self.samples)

        if not self.samples:
            raise ValueError(f"No images found in dataset folder: {root_dir}")

    def _filter_valid_samples(self, samples):
        valid_samples = []
        skipped = 0
        for image_path, label_idx in samples:
            try:
                with Image.open(image_path) as img:
                    img.verify()
                valid_samples.append((image_path, label_idx))
            except Exception:
                skipped += 1

        if skipped:
            print(f"[data_loader] Skipped {skipped} unreadable image(s) in {self.root_dir}")

        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def _normalise_image_dirs(image_dirs):
    if isinstance(image_dirs, (str, Path)):
        image_dirs = [image_dirs]

    image_dirs = [Path(image_dir) for image_dir in image_dirs]
    if not image_dirs:
        raise ValueError("At least one dataset directory is required.")

    missing_dirs = [str(image_dir) for image_dir in image_dirs if not image_dir.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Dataset directory not found: {', '.join(missing_dirs)}")

    return image_dirs


def _build_split_dataset(image_dirs, split_name):
    datasets = []
    for image_dir in image_dirs:
        split_dir = image_dir / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing {split_name!r} split in dataset: {image_dir}")

        dataset = RealFakeDataset(split_dir)
        datasets.append(dataset)
        print(f"[data_loader] {split_name}: {image_dir} -> {len(dataset)} images")

    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_dataloaders(batch_size=16, image_dirs=("datasets/real_vs_fake/real-vs-fake",)):
    """Build loaders from one or more datasets with train/valid/test/fake/real folders.

    Each input dataset must use the same label folders: ``fake`` is class 0 and
    ``real`` is class 1. Matching splits are concatenated so every batch is
    sampled from the combined training set.
    """
    image_dirs = _normalise_image_dirs(image_dirs)

    train_ds = _build_split_dataset(image_dirs, "train")
    valid_ds = _build_split_dataset(image_dirs, "valid")
    test_ds = _build_split_dataset(image_dirs, "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    return train_loader, valid_loader, test_loader
