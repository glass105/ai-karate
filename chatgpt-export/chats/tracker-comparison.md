# Tracker Comparison

Source: https://chatgpt.com/g/g-p-6a1b2af572ec8191922bad8b079f2496-ai-for-karate/c/6a1ca768-17f0-83ea-b075-524dbb781c2a

Tracker	Use case
ByteTrack	vs BoT-SORT

I’ll frame this around your karate prototype: two fighters, identity retention, occlusion, and strike-count reliability.

Thought for 4s

For your karate fighter tracking prototype, use this rule:

Tracker	Best use case	Why
ByteTrack	First prototype / real-time tracking / simpler camera angle	Fast, simple, works well when both fighters stay visible and do not overlap too much. Good starting point for punch/kick counting.
BoT-SORT	Better fighter identity retention	Better when fighters cross, clinch, overlap, or briefly block each other. Uses stronger association logic and can use appearance/ReID features. 
arXiv
+1
My recommendation

Start with ByteTrack first.

Use it to prove the pipeline:

YOLO pose detection → tracker ID → left/right fighter identity → keypoints → punch/kick logic → counts per fighter

Then move to BoT-SORT when you see ID-switching, for example:

Gabriel starts as ID 1
fighters cross
Gabriel becomes ID 2
strike counts get mixed up

That is where BoT-SORT is likely better, especially with ReID enabled. Ultralytics supports both trackers in its tracking workflow, including bytetrack.yaml and botsort.yaml. 
Ultralytics Docs

For your karate dojo use case
Scenario	Better choice
Single fixed camera	ByteTrack
Need fastest FPS	ByteTrack
Two fighters mostly separated	ByteTrack
Fighters crossing/clinching	BoT-SORT
Need Gabriel’s identity preserved across movement	BoT-SORT
Strike-count accuracy depends on fighter ID	BoT-SORT
Cloud GPU prototype on RunPod	Start ByteTrack, then test BoT-SORT
Practical answer

For your prototype:

Python
Run
tracker="bytetrack.yaml"

Then for the more serious version:

Python
Run
tracker="botsort.yaml"

Best path:

Phase 1: YOLO pose + ByteTrack + basic punch/kick counting
Phase 2: Switch to BoT-SORT if IDs swap
Phase 3: Add manual fighter selection: “Gabriel starts on left/right”
Phase 4: Add correction logic if tracker ID changes mid-round

For karate, BoT-SORT is probably the better final tracker, but ByteTrack is the better first implementation because it is faster and easier to debug.

Sources
Ultralytics vs  YOLO11-pose
Thought for a few seconds

Ultralytics is the framework/company/library.

YOLO11-pose is one specific pose-estimation model inside the Ultralytics ecosystem.

So it is not really:

Ultralytics vs YOLO11-pose

It is more like:

Ultralytics = the toolset
YOLO11-pose = the model you run inside that toolset
For your karate prototype
Item	What it means	Use in your project
Ultralytics	Python package / CLI / training / tracking / export framework	Runs detection, pose, tracking, training, export
YOLO11-pose	Human keypoint model	Finds shoulders, elbows, wrists, hips, knees, ankles
ByteTrack / BoT-SORT	Trackers supported through Ultralytics tracking mode	Keeps fighter identity across frames
Custom strike logic	Your own code	Counts punches, kicks, movement, guard, distance

Ultralytics’ pose models return body keypoints, and their docs show pose models using the -pose suffix with keypoint outputs such as result.keypoints. The default human pose setup uses 17 keypoints trained on COCO keypoints. 
Ultralytics Docs

Recommended stack

For your prototype, I would use:

Python
Run
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

results = model.track(
    source="fight_video.mp4",
    tracker="bytetrack.yaml",
    persist=True
)

Then later test:

Python
Run
results = model.track(
    source="fight_video.mp4",
    tracker="botsort.yaml",
    persist=True
)
Which YOLO11-pose size?
Model	Best for	Recommendation
yolo11n-pose.pt	Fastest, cheapest GPU use	Start here
yolo11s-pose.pt	Better accuracy, still fast	Good prototype upgrade
yolo11m-pose.pt	Better keypoints, more GPU	Better for final testing
yolo11l/x-pose.pt	Highest accuracy, slower	Only if RunPod GPU budget allows

YOLO11 supports pose estimation as one of its official tasks, along with detection, segmentation, classification, and oriented bounding boxes. Ultralytics describes YOLO11 as optimized for efficiency and speed with improved feature extraction. 
Ultralytics Docs

My recommendation for you

Use:

Ultralytics + YOLO11-pose + ByteTrack first

Then move to:

Ultralytics + YOLO11-pose + BoT-SORT

For karate, the key is not just the model. The real value will come from your custom logic:

fighter ID → body keypoints → arm/leg extension → strike event → punch/kick count

So the clean answer is:

Use Ultralytics as the platform, YOLO11-pose as the pose model, and ByteTrack/BoT-SORT as the fighter identity tracker.

Sources

Thinking
ChatGPT can make mistakes. Check important info.
