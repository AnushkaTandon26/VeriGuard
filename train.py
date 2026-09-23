import argparse
import copy
import json
from pathlib import Path

import h5py
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from data_loader import build_dataloaders
from model import build_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_IMAGE_DIR = "datasets/real_vs_fake/real-vs-fake"
DEFAULT_IMAGE_DIRS = [
    DEFAULT_IMAGE_DIR,
    "datasets/dffmd_images",
]
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 2e-5
DEFAULT_H5_OUTPUT = "models/fa3_real_vs_fake_dffmd_trained_model.h5"


def sanitize_name(value):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def build_run_name(image_dirs, run_name):
    if run_name:
        return sanitize_name(run_name)
    if len(image_dirs) == 1:
        return f"clip_{sanitize_name(Path(image_dirs[0]).name or 'dataset')}"
    return "fa3_clip_combined"


def export_model_h5(model, model_name, output_path):
    """Save CLIP weights and metadata in an HDF5 file.

    The FA³ checkpoint directory is also retained for native PyTorch loading.
    The H5 file stores the same PyTorch state dictionary for delivery or archival.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["format"] = "PyTorch FA3-CLIP state_dict"
        h5_file.attrs["model_name"] = model_name
        h5_file.attrs["id2label"] = json.dumps(model.config.id2label)
        h5_file.create_dataset("config_json", data=model.config.to_json_string())

        weights_group = h5_file.create_group("state_dict")
        for parameter_name, tensor in model.state_dict().items():
            tensor = tensor.detach().cpu()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            weights_group.create_dataset(parameter_name, data=tensor.numpy())

    print(f"Saved H5 model weights to {output_path}")


def calculate_metrics(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["fake", "real"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["fake", "real"],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": report_text,
        "classification_report_dict": report_dict,
    }


def train(args):
    run_name = build_run_name(args.image_dirs, args.run_name)
    model_dir = Path(args.model_dir) if args.model_dir else Path("models") / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file) if args.metrics_file else outputs_dir / f"{run_name}_metrics.json"

    print("Datasets:")
    for image_dir in args.image_dirs:
        print(f"  - {image_dir}")
    print(f"Run name: {run_name}")
    print(f"Model dir: {model_dir}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")

    train_loader, valid_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        image_dirs=args.image_dirs,
    )
    print(
        "Combined split sizes: "
        f"train={len(train_loader.dataset)} | "
        f"valid={len(valid_loader.dataset)} | "
        f"test={len(test_loader.dataset)}"
    )
    model, processor = build_model(model_name=args.model_name, num_labels=2)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for images, labels in train_loader:
            inputs = processor(images=list(images), return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, labels in valid_loader:
                inputs = processor(images=list(images), return_tensors="pt", padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                label_tensors = labels.to(DEVICE)

                outputs = model(**inputs, labels=label_tensors)
                val_losses.append(outputs.loss.item())

                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(labels.tolist())

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"Epoch {epoch + 1} | train_loss={sum(train_losses)/len(train_losses):.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            model.save_pretrained(model_dir)
            processor.save_pretrained(model_dir)
            print(f"Saved best checkpoint to {model_dir}")

    # Test and export the best validation checkpoint, not merely the final epoch.
    if best_state is None:
        raise RuntimeError("No best checkpoint was created; use at least one training epoch.")
    model.load_state_dict(best_state)
    model.to(DEVICE)

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            inputs = processor(images=list(images), return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = calculate_metrics(y_true, y_pred)

    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Test Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Test F1 score (macro): {metrics['f1_macro']:.4f}")
    print(f"Test Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Test Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"Test F1 score (weighted): {metrics['f1_weighted']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("\nClassification Report:\n")
    print(metrics["classification_report"])

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved model to {model_dir}")
    export_model_h5(model, args.model_name, args.h5_output)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP binary classifier on fake/real datasets.")
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        default=DEFAULT_IMAGE_DIRS,
        help="One or more dataset roots containing train/valid/test/fake/real folders.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face CLIP model id.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--run-name", default=None, help="Optional run name used for output naming.")
    parser.add_argument("--model-dir", default=None, help="Optional explicit directory to save model.")
    parser.add_argument("--metrics-file", default=None, help="Optional explicit path for metrics JSON.")
    parser.add_argument(
        "--h5-output",
        default=DEFAULT_H5_OUTPUT,
        help="H5 file where the best trained CLIP weights are saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
