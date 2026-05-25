---
title: Face Redaction Tool
emoji: 🕶️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
---

# Face Redaction Tool : LH2 Data Labs

Tool link: https://huggingface.co/spaces/Harsh-a-Vardhan/face-redaction-tool 

Removes facial PII from images by covering the **eyes** and **mouth** with
a semi-translucent black box, while keeping the rest of the image usable
for AI training. Handles multiple faces per image. Built as part of the Founder's Office assignment.

## Video Walkthrough

**Loom Link: https://www.loom.com/share/4776fb04955347b4a68a50f13a54e0f3**

## How it works

- **Detection:** YuNet, a compact deep-learning face detector (OpenCV
  DNN). It returns a face box plus five landmarks : both eyes, nose, and
  the two mouth corners.
- **Redaction:** The eye and mouth boxes are placed directly on those
  landmarks, so they track each face's real geometry. Boxes are fixed at
  0.92 opacity — strong enough to remove identity, with slight
  translucency. Opacity is intentionally not adjustable.
- **Fallback:** on strongly tilted heads, landmarks can be unreliable; the
  tool detects this and falls back to face-box bands so no face is missed.
- **Debug mode:** draws detection boxes as coloured outlines for checking.

## Files

- `app.py` — Gradio web interface
- `redactor.py` — detection and redaction engine
- `requirements.txt` — dependencies
- `face_detection_yunet_2023mar.onnx` — YuNet model
- `sample_faces/` — gallery images

## Run locally

```
pip install -r requirements.txt
python app.py
```

The model file must sit next to `redactor.py`.

## Design notes

Started with OpenCV Haar cascades (no model download, simple deploy), but
they produced false detections on busy backgrounds and struggled with
hair over the face. Switched to YuNet, which is robust on real-world
photos and provides landmarks for precise box placement. Later
refinements: removed an opacity slider (a privacy tool shouldn't let the
redaction be weakened), tuned box sizes so the mouth box covers the lips
without reaching the nose or chin and added the tilted-face fallback.
