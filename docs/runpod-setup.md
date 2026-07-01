# RunPod Development Setup

Use a GPU Pod for the prototype. Serverless can wait until the video pipeline
works reliably.

## 1. Keep Secrets Out Of Git

Create a RunPod API key in your RunPod account and configure `runpodctl` on the
machine where you will deploy. Do not add the API key to this repository.

Copy `.env.example` into your shell environment and set:

```bash
export RUNPOD_IMAGE=YOUR_DOCKERHUB_USERNAME/ai-karate:latest
export RUNPOD_GPU_ID="NVIDIA GeForce RTX 4090"
export RUNPOD_NETWORK_VOLUME_ID=YOUR_VOLUME_ID
```

Keep `RUNPOD_NETWORK_VOLUME_ID` set to the same network volume ID when
recreating a Pod. The deployment helpers automatically attach that existing
volume at `/workspace`.

## 2. Build And Publish The Image

Build this image on a machine with Docker and push it to a registry:

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/ai-karate:latest .
docker push YOUR_DOCKERHUB_USERNAME/ai-karate:latest
```

The Dockerfile extends RunPod's PyTorch image and preserves its default
interactive startup behavior for SSH and Jupyter development.

## 3. Create Persistent Storage

For an initial manual deployment, create a network volume in the RunPod console
and attach it while deploying the Pod. RunPod typically mounts it at
`/workspace`.

The CLI equivalent is:

```bash
runpodctl datacenter list
runpodctl network-volume create \
  --name ai-karate-data \
  --size 100 \
  --data-center-id YOUR_DATACENTER_ID
runpodctl network-volume list
```

Network volumes must be attached while the Pod is created. They cannot be
attached or detached later without deleting the Pod.

## 4. Create The Development Pod

Review the defaults in `scripts/create_runpod_pod.sh`, then run:

```bash
bash scripts/create_runpod_pod.sh
```

On Windows, double-click `scripts/create_runpod_bytetrack.cmd` or
`scripts/create_runpod_botsort.cmd`. Each button creates the matching tracker
Pod without editing `.env`. The Pod name includes its storage type and tracker,
for example `ai-karate-dev-disposable-bytetrack`.

The helper creates a Secure Cloud GPU Pod, exposes SSH and Jupyter ports, and
adds an automatic termination timer to limit accidental spend. It requires
`RUNPOD_NETWORK_VOLUME_ID` by default so that recreating a Pod reconnects the
existing `/workspace` data.

For an intentionally disposable Pod volume, explicitly set:

```bash
export RUNPOD_ALLOW_EPHEMERAL_VOLUME=true
```

## 5. Prepare The Workspace

From the Pod terminal:

```bash
cd /workspace
git clone YOUR_GITHUB_REPO_URL ai-karate
cd ai-karate
bash scripts/bootstrap_workspace.sh /workspace/ai-karate
```

Upload a short clip to:

```text
/workspace/ai-karate/videos/input/fight1.mp4
```

Run the first analysis:

```bash
cd /workspace/ai-karate
python -m src.analyze_video \
  --input videos/input/fight1.mp4 \
  --output-dir videos/output \
  --model yolo11s-pose.pt \
  --tracker bytetrack.yaml \
  --fighter-a-name Gabriel \
  --fighter-a-start left
```

Then compare identity retention:

```bash
python -m src.analyze_video \
  --input videos/input/fight1.mp4 \
  --output-dir videos/output/botsort \
  --model yolo11s-pose.pt \
  --tracker botsort.yaml \
  --fighter-a-name Gabriel \
  --fighter-a-start left
```

## 6. Review Before Expanding

Inspect the annotated video, track CSV, and JSON summary. The next engineering
step is identity recovery for tracker swaps, followed by calibration against
manually reviewed karate clips.
