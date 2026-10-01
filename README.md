# GWU Praxis

Research on adaptive camera resolution and YOLO model selection under compute constraints.

## Training

YOLOv8n, YOLOv8s, and YOLOv8m were trained in Google Colab using the baseline three-stage workflow:

1. Generic digit training: 11,000 training / 2,200 validation / 2,200 test images.
2. Field digit crop fine-tuning: 13,104 training / 2,808 validation / 6,864 test images.
3. Full-board fine-tuning: 336 training / 72 validation / 176 test images.

The final checkpoints are stored in models using Git LFS.
YOLOv8l training and evaluation are pending.

## Evaluation

Evaluated 255 capture sets from September 23–28, 2026.
Three models across four native capture resolutions produced 3,060 evaluations.

Inference settings: imgsz=1280, confidence=0.25, IoU=0.45.
Passing criterion: at least 34 of 40 board digits correct.

Template alignment with a feature-based fallback produced zero image/alignment rejections.

At 2028x1520:
- YOLOv8s passed 172/255 captures (67.5%).
- YOLOv8m passed 177/255 captures (69.4%).
- Either model passed 214/255 captures (83.9%).

Across all 12 configurations, at least one passed on 240/255 captures (94.1%).
This is a hindsight upper bound, not the performance of a trained adaptive selector.

Evaluation timings were measured on the Windows desktop CPU and exclude camera capture and alignment.

## Contents

- models: final n/s/m checkpoints
- evaluation: alignment and scoring script
- results/20260930: evaluation scores and summaries
- docs: historical baseline training settings

The evaluation script currently uses local C:\PraxisData paths.
Raw image datasets are maintained separately.
