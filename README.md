# VeriGuard deepfake detector

This repository contains a local web application for screening image and voice
deepfakes. Uploaded files are processed in memory and are not stored by the
application.

## Run the detector

The frontend combines two trained models:

- FA³-CLIP DFFMD for JPG, PNG, and WebP face images.
- CNN-Conformer for WAV, MP3, OGG, WebM, M4A, and FLAC voice clips. Its trained
  checkpoint is included at `models/audio/cnn_conformer.h5`.

Install the audio dependencies once before the first voice scan:

```bash
./venv/bin/pip install -r requirements.txt
```

```bash
./venv/bin/python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The
checkpoint is loaded only when you submit the first image, so that first
analysis can take longer. Stop the local server with `Ctrl+C`.

Useful options:

```bash
./venv/bin/python app.py --port 8080
./venv/bin/python app.py --host 0.0.0.0 --port 8000
```

The JSON endpoints are `POST /api/predict` with a multipart field named
`image`, and `POST /api/predict-audio` with a multipart field named `audio`.

## Train the image model

The training script implements the FA³-CLIP components from the paper:
high-pass FFT frequency cues, spatial/frequency fusion, frequency-aware CNN
features, and conditional attack-agnostic live/fake prompt biases. It combines
matching `train`, `valid`, and `test` splits from both prepared datasets using
the CLIP backbone `openai/clip-vit-base-patch32`. By default it uses:

- `datasets/real_vs_fake/real-vs-fake`
- `datasets/dffmd_images`

Run the combined training job with:

```bash
./venv/bin/python train.py --epochs 1 --batch-size 8 --run-name fa3_clip_combined
```

This writes the best FA³-CLIP checkpoint and evaluation metrics to local paths
under `models/` and `outputs/`. Those generated files are intentionally ignored
by Git because model checkpoints and datasets are too large for a normal GitHub
repository.

Use a full pretrained CLIP checkpoint for `--model-name`; a
`CLIPForImageClassification` directory is not a full vision-language CLIP
checkpoint because it does not contain the CLIP text encoder.

To explicitly choose the datasets, use:

```bash
./venv/bin/python train.py \
  --image-dirs datasets/real_vs_fake/real-vs-fake datasets/dffmd_images \
  --epochs 1 --batch-size 8 --run-name fa3_clip_combined
```

## Prepare the Artifact Faces dataset

The 30 GB Artifact archive contains many unrelated datasets. The preparation
script extracts only FFHQ as `real` and FaceSynthetics as `fake`, then creates
the six folders required by the trainer:

```bash
./venv/bin/python prepare_artifact_faces.py
```

After preparation, train it separately with:

```bash
./venv/bin/python train.py \
  --image-dirs datasets/artifact_faces \
  --epochs 1 --batch-size 16 \
  --run-name fa3_artifact_faces \
  --h5-output models/fa3_artifact_faces_trained_model.h5
```

## Repository scope

The repository keeps source code and the web interface only. Downloaded
datasets, archives, virtual environments, model checkpoints, caches, and
generated metrics remain local and are excluded by `.gitignore`. Place the
required runtime checkpoints at:

- `models/fa3_dffmd/`
- `models/audio/cnn_conformer.h5`

The training and preparation scripts are retained for reproducibility; they
expect the datasets described above to be available locally.
