import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


def create_output_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def save_history(history):
    """
    Save training history
    """

    history_df = pd.DataFrame(history.history)
    history_df.to_csv("outputs/history.csv", index=False)


def plot_accuracy(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig("outputs/accuracy.png")

    plt.close()


def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig("outputs/loss.png")

    plt.close()


def evaluate_model(model, test_dataset):

    predictions = model.predict(test_dataset)

    y_pred = (predictions > 0.5).astype(int)

    y_true = np.concatenate(
        [labels.numpy() for images, labels in test_dataset],
        axis=0
    ).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake", "Real"]
    )

    plt.figure(figsize=(6,6))

    disp.plot(cmap="Blues")

    plt.savefig("outputs/confusion_matrix.png")

    plt.close()

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Fake", "Real"]
    )

    with open(
        "outputs/classification_report.txt",
        "w"
    ) as f:
        f.write(report)

    print(report)