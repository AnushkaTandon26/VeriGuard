#!/usr/bin/env python3
"""Prepare FFHQ + FaceSynthetics from artifact-dataset.zip for FA3-CLIP."""

import argparse
import random
import zipfile
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCES = {
    "real": "ffhq/ffhq/images/",
    "fake": "face_synthetics/facesyn/images/",
}


def split_items(items, train_ratio, valid_ratio, seed):
    items = list(items)
    random.Random(seed).shuffle(items)
    n_train = int(len(items) * train_ratio)
    n_valid = int(len(items) * valid_ratio)
    return (
        items[:n_train],
        items[n_train : n_train + n_valid],
        items[n_train + n_valid :],
    )


def main(args):
    zip_path = Path(args.zip_path)
    output_root = Path(args.output_root)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Choose another --output-root."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        for label, prefix in SOURCES.items():
            image_members = [
                name
                for name in members
                if name.startswith(prefix) and Path(name).suffix.lower() in IMAGE_EXTENSIONS
            ]
            if args.max_images_per_class is not None:
                image_members = image_members[: args.max_images_per_class]
            if not image_members:
                raise ValueError(f"No images found under {prefix} in {zip_path}")

            splits = split_items(image_members, args.train_ratio, args.valid_ratio, args.seed)
            print(f"{label}: {len(image_members)} images")
            for split_name, split_members in zip(("train", "valid", "test"), splits):
                split_dir = output_root / split_name / label
                split_dir.mkdir(parents=True, exist_ok=True)
                for index, member_name in enumerate(
                    tqdm(split_members, desc=f"{split_name}/{label}"), start=1
                ):
                    # Prefixing with the class and a counter avoids collisions.
                    suffix = Path(member_name).suffix.lower() or ".jpg"
                    output_path = split_dir / f"{label}_{index:07d}{suffix}"
                    with archive.open(member_name) as source, output_path.open("wb") as target:
                        target.write(source.read())

    print(f"\nPrepared FA3-CLIP dataset at: {output_root}")
    print("Classes: fake=FaceSynthetics, real=FFHQ")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        default="/home/CL502-31/Downloads/artifact-dataset.zip",
        help="Path to artifact-dataset.zip.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/artifact_faces",
        help="Output root with train/valid/test/fake/real folders.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Optional cap per class for a quick test; omit for the full subsets.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
