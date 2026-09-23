#!/usr/bin/env python3

import argparse
import random
import zipfile
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm


VALID_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def ensure_extracted(zip_path: Path, extract_root: Path) -> Path:
    source_root = extract_root / "deepfake-face-mask-dataset-dffmd"
    if source_root.exists():
        return source_root

    print(f"Extracting {zip_path} to {extract_root} ...")
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(source_root)
    return source_root


def gather_videos(source_root: Path):
    fake_videos = []
    real_videos = []

    for video_path in source_root.rglob("*"):
        if video_path.suffix.lower() not in VALID_VIDEO_EXTS:
            continue
        parts = {p.lower() for p in video_path.parts}
        if "fake" in parts:
            fake_videos.append(video_path)
        elif "real" in parts:
            real_videos.append(video_path)

    if not fake_videos or not real_videos:
        raise ValueError("Could not find both Fake and Real videos in extracted dataset")

    return sorted(fake_videos), sorted(real_videos)


def split_items(items, train_ratio, valid_ratio, seed):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid

    train_items = items[:n_train]
    valid_items = items[n_train:n_train + n_valid]
    test_items = items[n_train + n_valid:n_train + n_valid + n_test]
    return train_items, valid_items, test_items


def save_first_frame(video_path: Path, out_path: Path):
    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()

    image = Image.fromarray(frame).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="JPEG", quality=95)


def convert_split(videos, split_name, label_name, out_root: Path, skipped):
    out_dir = out_root / split_name / label_name
    for idx, video_path in enumerate(tqdm(videos, desc=f"{split_name}/{label_name}"), start=1):
        out_name = f"{video_path.stem}_{idx:06d}.jpg"
        out_path = out_dir / out_name
        try:
            save_first_frame(video_path, out_path)
        except Exception:
            skipped.append(str(video_path))


def main():
    parser = argparse.ArgumentParser(description="Prepare DFFMD video dataset into image folders for CLIP training.")
    parser.add_argument(
        "--zip-path",
        default="datasets/deepfake-face-mask-dataset-dffmd.zip",
        help="Path to DFFMD zip file.",
    )
    parser.add_argument(
        "--extract-root",
        default="datasets",
        help="Directory where the zip will be extracted.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/dffmd_images",
        help="Output dataset root with train/valid/test/fake/real folders.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=None,
        help="Optional cap per class for quick tests.",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    extract_root = Path(args.extract_root)
    out_root = Path(args.output_root)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    if args.train_ratio <= 0 or args.valid_ratio <= 0 or (args.train_ratio + args.valid_ratio) >= 1:
        raise ValueError("Use ratios where train_ratio > 0, valid_ratio > 0, and train+valid < 1")

    source_root = ensure_extracted(zip_path, extract_root)
    fake_videos, real_videos = gather_videos(source_root)

    if args.max_videos_per_class is not None:
        fake_videos = fake_videos[: args.max_videos_per_class]
        real_videos = real_videos[: args.max_videos_per_class]

    print(f"Found fake videos: {len(fake_videos)}")
    print(f"Found real videos: {len(real_videos)}")

    fake_train, fake_valid, fake_test = split_items(fake_videos, args.train_ratio, args.valid_ratio, args.seed)
    real_train, real_valid, real_test = split_items(real_videos, args.train_ratio, args.valid_ratio, args.seed)

    if out_root.exists():
        print(f"Output root already exists: {out_root}")
        print("New files will be written into existing folders.")

    skipped = []

    convert_split(fake_train, "train", "fake", out_root, skipped)
    convert_split(fake_valid, "valid", "fake", out_root, skipped)
    convert_split(fake_test, "test", "fake", out_root, skipped)

    convert_split(real_train, "train", "real", out_root, skipped)
    convert_split(real_valid, "valid", "real", out_root, skipped)
    convert_split(real_test, "test", "real", out_root, skipped)

    print("\nDataset ready:")
    print(f"  {out_root}")
    print("Train/Valid/Test folders created with fake/real classes.")

    if skipped:
        skipped_path = out_root / "skipped_videos.txt"
        skipped_path.write_text("\n".join(skipped))
        print(f"Skipped videos: {len(skipped)}")
        print(f"Saved skipped list to: {skipped_path}")


if __name__ == "__main__":
    main()
