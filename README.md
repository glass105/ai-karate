# AI Karate

Starter repository for a karate dojo video-analysis prototype.

## First Goal

Analyze one short fight video and produce:

1. An annotated MP4 with pose overlays.
2. A per-frame CSV with track IDs, bounding boxes, keypoints, and estimated actions.
3. A JSON report with rough punch and kick counts per fighter.

The first version deliberately uses simple heuristics. It is a measurement
baseline for tuning with real karate footage, not a scoring system.

## Pipeline

```text
fight video
  -> Ultralytics YOLO11 pose
  -> ByteTrack or BoT-SORT
  -> starting-side fighter identity mapping
  -> pose keypoints
  -> punch/kick candidate heuristics
  -> annotated MP4 + CSV + JSON
```

Start with ByteTrack because it is easier to debug. Compare BoT-SORT when
crossings, clinches, or occlusions cause identity swaps.

## Repository Layout

```text
src/                     Python analysis pipeline
tests/                   Unit tests for local logic
scripts/                 RunPod setup helpers
docs/runpod-setup.md     Manual RunPod setup guide
review/fighters/         Lightweight fighter review summaries by fighter name
videos/input/            Input videos, ignored by Git
videos/output/           Generated artifacts, ignored by Git
models/                  Downloaded models, ignored by Git
chatgpt-export/          Original planning conversations
```

## Fighter Review Outputs

Small review artifacts for each fighter should be committed under:

```text
review/fighters/<fighter-name>/latest/
```

Example:

```text
review/fighters/gabriel/latest/manifest.json
review/fighters/gabriel/latest/summary.json
```

Use lowercase folder names for fighters, such as `gabriel` or `john-smith`. Keep heavy generated artifacts out of GitHub, including annotated videos, large CSV files, frame dumps, logs, and model weights. Those should remain in `videos/output/`, RunPod storage, or another artifact store.

Older or experimental review summaries can be preserved under:

```text
review/fighters/<fighter-name>/runs/<run-label>/
```

See [review/fighters/README.md](review/fighters/README.md) for the review directory policy.

## Run Locally

Create a Python 3.11 environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Analyze a clip:

```bash
python -m src.analyze_video \
  --input videos/input/fight1.mp4 \
  --output-dir videos/output \
  --model yolo11s-pose.pt \
  --tracker bytetrack.yaml \
  --fighter-a-name Gabriel \
  --fighter-a-start left
```

Switch to `--tracker botsort.yaml` for an identity-retention comparison.

## Start On RunPod

The first RunPod setup is documented in [docs/runpod-setup.md](docs/runpod-setup.md).
It uses a GPU Pod and persistent storage mounted at `/workspace`. No RunPod
credentials belong in this repository.

## Planning Archive

The recovered ChatGPT conversations are indexed in
[chatgpt-export/README.md](chatgpt-export/README.md).
