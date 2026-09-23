#!/usr/bin/env python3
"""Evaluate a saved FA3-CLIP H5 model on a dataset test split."""

import argparse
import json
from pathlib import Path

import h5py
import torch

from data_loader import build_dataloaders
from model import build_model
from train import calculate_metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_h5_model(h5_path):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5_file:
        model_name = h5_file.attrs["model_name"]
        if isinstance(model_name, bytes):
            model_name = model_name.decode()
        state = {
            name: torch.as_tensor(dataset[()])
            for name, dataset in h5_file["state_dict"].items()
        }

    model, processor = build_model(model_name=model_name, num_labels=2)
    model.load_state_dict(state)
    return model.to(DEVICE), processor


def main(args):
    model, processor = load_h5_model(args.model_h5)
    _, _, test_loader = build_dataloaders(batch_size=args.batch_size, image_dirs=[args.image_dir])

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            inputs = processor(images=list(images), return_tensors="pt", padding=True)
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
            predictions = model(**inputs).logits.argmax(dim=-1)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print(metrics["classification_report"])

    if args.metrics_file:
        output_path = Path(args.metrics_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved evaluation metrics to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an FA3-CLIP H5 model.")
    parser.add_argument("--model-h5", required=True, help="Path to the saved FA3-CLIP .h5 file.")
    parser.add_argument("--image-dir", required=True, help="Dataset root containing train/valid/test splits.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--metrics-file", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
