# Next Steps Setup Guide

Source: https://chatgpt.com/g/g-p-6a1b2af572ec8191922bad8b079f2496-ai-for-karate/c/6a1d827d-1dbc-83ea-9bb0-59775962b0bf

Next steps

What to do next. 

runpod account is created
github is connected to codex

what next

I’ll lay this out as a practical build sequence: repo setup, RunPod template/pod setup, then what to ask Codex to implement first so you get a working prototype before adding intelligence.

Thought for 10s

Here is what I would do next.

1. Create the GitHub repo structure first

Make sure the repo connected to Codex has a clean starting structure:

The first goal is not perfection. The first goal is a working pipeline:

Ultralytics already supports pose models and tracking with BoT-SORT / ByteTrack, and BoT-SORT is the default tracker in Ultralytics tracking mode. 
Ultralytics Docs
+1

2. Add the basic dependency file

Create requirements.txt:

Later you can add FastAPI, Streamlit, Gradio, or a dashboard.

3. Add a simple Dockerfile for RunPod

Start simple:

dockerfile

RunPod templates let you preconfigure Docker images, storage, exposed ports, and volume mount paths. Persistent volume storage is important because /workspace can preserve models, videos, and outputs between pod restarts. 
Runpod Documentation
+1

4. Create the RunPod pod manually first

Do not over-automate yet.

In RunPod:

Choose a GPU pod.

Use a PyTorch/CUDA template or your custom Dockerfile later.

Add persistent volume.

Mount it to:

/workspace

Open SSH or JupyterLab.

Clone your GitHub repo:

Bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/karate-fighter-analysis.git
cd karate-fighter-analysis
pip install -r requirements.txt

RunPod supports connecting to pods through SSH, web proxy, JupyterLab, and IDE-style workflows. 
Runpod Documentation

5. Test YOLO pose tracking before strike counting

Run this first from the pod:

Bash
yolo track \
  model=yolo11s-pose.pt \
  source=videos/input/fight1.mp4 \
  tracker=botsort.yaml \
  save=True

Also test ByteTrack:

Bash
yolo track \
  model=yolo11s-pose.pt \
  source=videos/input/fight1.mp4 \
  tracker=bytetrack.yaml \
  save=True

ByteTrack is usually faster and simpler; BoT-SORT is usually better when fighters cross, occlude, or change direction because it has stronger tracking behavior. Ultralytics supports both via tracker=botsort.yaml and tracker=bytetrack.yaml. 
Ultralytics Academy
+1

For your karate use case, start with:

BoT-SORT first
ByteTrack as fallback comparison
6. Give Codex a very specific first task

Since GitHub is already connected to Codex, the next move is to give Codex one contained task. Codex can work on your repo, write features, fix bugs, and propose pull requests. 
OpenAI Developers
+1

Use this prompt:

Create the first working prototype for karate fighter video analysis.

Repo goal:
Use Ultralytics YOLO pose tracking to analyze a fight video and produce:
1. Annotated output video
2. CSV with per-frame fighter tracking data
3. Per-interval summary of estimated punches and kicks

Implementation requirements:
- Main entry point: src/analyze_video.py
- CLI arguments:
  --input videos/input/fight1.mp4
  --output-dir videos/output
  --model yolo11s-pose.pt
  --tracker botsort.yaml
  --fighter-a-name Gabriel
  --fighter-a-start left
  --interval-seconds 60
- Use Ultralytics model.track with persist=True
- Save annotated video
- Save CSV with frame number, timestamp, track_id, bbox, keypoints, estimated action
- Add basic heuristic strike counting:
  - punch estimate from wrist extension relative to shoulder/hip line
  - kick estimate from ankle/knee extension relative to hip
  - avoid double-counting using cooldown frames
- Add README instructions
- Add a small test command
- Do not build a web UI yet

That is your first useful milestone.

7. Then give Codex a second task: fighter identity locking

After the first script works, ask Codex this:

Improve fighter identity handling.

Problem:
The user wants to track a specific fighter, for example Gabriel, who starts on the left side with white gloves.

Add logic to:
1. Assign Fighter A based on --fighter-a-start left/right during the first N frames
2. Keep Fighter A mapped to the most consistent track_id
3. Handle temporary lost tracks
4. Add a warning when identity confidence is low
5. Save fighter_identity_debug.csv
6. Update README with examples

This matters because strike counts are useless if the tracker swaps fighters.

8. Then build the output report

Third Codex task:

Add reporting.

Create src/report.py that generates:
1. interval_summary.csv
2. fighter_summary.json
3. markdown report

Report should include:
- punches per interval
- kicks per interval
- movement distance estimate
- forward/backward movement estimate
- stance-switch or direction-change estimate if possible
- notes about confidence and limitations

Your report should look like:

0:00–1:00
Gabriel: 6 punches, 2 kicks, high mobility, 1 possible identity uncertainty

1:00–2:00
Gabriel: 4 punches, 3 kicks, moderate mobility
9. Do not build the full dashboard yet

Recommended order:

Phase 1: Command-line video analyzer
Phase 2: Fighter identity stability
Phase 3: CSV/Markdown reports
Phase 4: Annotated video overlays
Phase 5: Simple Streamlit or Gradio dashboard
Phase 6: Expert comparison model

The expert comparison piece should come later because you first need reliable pose, tracking, and event counting.

10. What I would do today

Today’s exact checklist:

1. Create repo structure
2. Add requirements.txt
3. Add Dockerfile
4. Launch RunPod GPU pod
5. Clone repo into /workspace
6. Upload one fight video to videos/input/
7. Run YOLO pose tracking manually
8. Ask Codex to create src/analyze_video.py
9. Run the script on one 60-second clip
10. Review annotated video and CSV

Your immediate next command on RunPod should be:

Bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/karate-fighter-analysis.git
cd karate-fighter-analysis
pip install -r requirements.txt
mkdir -p videos/input videos/output models runs

Then upload fight1.mp4 into:

videos/input/

And test:

Bash
yolo track model=yolo11s-p-pose.pt source=videos/input/fight1.mp4 tracker=botsort.yaml save=True

Correction: if using YOLO11 pose, the model name should normally be:

Bash
yolo track model=yolo11s-pose.pt source=videos/input/fight1.mp4 tracker=botsort.yaml save=True

That is the next real milestone: prove the pod can run pose tracking on one fight video.

Sources

Thinking
ChatGPT can make mistakes. Check important info.
