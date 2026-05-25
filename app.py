"""
Face Redaction Tool -- LH2 Data Labs
Founder's Office Assignment L2

A simple web tool that removes facial identity (PII) from images by
overlaying semi-translucent black boxes over the eyes and mouth, while
keeping the rest of the image usable for downstream AI training.

Detection uses the YuNet deep-learning face detector (via OpenCV), which
is robust to busy backgrounds and hair, and returns eye/mouth landmarks
directly so redaction boxes land on the actual features.

Run locally:   python app.py
Hosted on:     Hugging Face Spaces (Gradio SDK)
"""

import os
import glob
import cv2
import numpy as np
import gradio as gr

from redactor import redact_image, detect_faces, _rotated_rect_points

SAMPLE_DIR = "sample_faces"


# --------------------------------------------------------------------------
# Debug overlay: draw detection boxes as coloured outlines instead of
# solid redaction, so box placement can be visually verified on real faces.
# --------------------------------------------------------------------------
def debug_overlay(image_bgr, main_only):
    img = image_bgr.copy()
    faces = detect_faces(img, main_only=main_only)
    for f in faces:
        # Face box -- blue (axis-aligned)
        cv2.rectangle(img, f["face_box"][:2], f["face_box"][2:],
                      (255, 120, 0), 2)
        # Eye band -- green, mouth band -- red (rotated outlines)
        for region, colour in ((f["eye_box"], (0, 220, 0)),
                               (f["mouth_box"], (0, 0, 230))):
            pts = _rotated_rect_points(region["center"], region["size"],
                                       region["angle"])
            cv2.polylines(img, [pts], isClosed=True, color=colour,
                          thickness=2)
        label = f"{f['score']:.2f} {f.get('method', '')}"
        fx, fy = f["face_box"][0], f["face_box"][1]
        cv2.putText(img, label, (fx, max(12, fy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2)
    return img, len(faces)


# --------------------------------------------------------------------------
# Main handler
# --------------------------------------------------------------------------
def process(image_rgb, main_only, debug):
    """Gradio passes images in as RGB numpy arrays.

    Redaction opacity is fixed (not user-adjustable): a privacy tool must
    not let the user weaken the redaction to where identity is visible.
    """
    if image_rgb is None:
        return None, "Please upload an image or pick a sample first."

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        if debug:
            out_bgr, n = debug_overlay(image_bgr, main_only)
            if n == 0:
                msg = ("Debug mode: no face detected. Try a clearer, "
                       "more front-facing photo.")
            else:
                msg = (f"Debug mode: {n} face(s) detected. "
                       "Blue = face, green = eye band, red = mouth band. "
                       "If a band is off, tell me which way to nudge it.")
        else:
            out_bgr, n = redact_image(image_bgr, main_only=main_only)
            if n == 0:
                msg = ("No face detected -- nothing redacted. Try a "
                       "clearer, more front-facing photo.")
            elif n == 1:
                msg = "Redacted 1 face: eyes and mouth covered."
            else:
                msg = f"Redacted {n} faces: eyes and mouth covered on each."
    except Exception as e:  # noqa: BLE001
        return None, f"Error: {e}"

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb, msg


def load_samples():
    """Collect sample images for the gallery."""
    if not os.path.isdir(SAMPLE_DIR):
        return []
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths += glob.glob(os.path.join(SAMPLE_DIR, ext))
    paths = [p for p in sorted(paths)
             if "_redacted" not in p and "_debug" not in p]
    return paths


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
with gr.Blocks(title="Face Redaction Tool - LH2 Data Labs") as demo:
    gr.Markdown(
        "# Face Redaction Tool\n"
        "**LH2 Data Labs - Founder's Office Assignment L2**\n\n"
        "Upload a photo of a person (JPG or PNG). The tool detects the "
        "face and covers the **eyes and mouth** with a semi-translucent "
        "black box -- removing identity (PII) while keeping the rest of "
        "the image usable for downstream AI training.\n\n"
        "Detection uses the YuNet deep-learning face detector. Turn on "
        "**Debug mode** to see exactly where each region is detected."
    )

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Input image (JPG / PNG)")
            main_only = gr.Checkbox(
                value=False,
                label="Redact main subject only "
                      "(default: redact every face in the image)")
            debug = gr.Checkbox(
                value=False,
                label="Debug mode (show detection boxes, not redaction)")
            btn = gr.Button("Redact face", variant="primary")

            samples = load_samples()
            if samples:
                gr.Markdown("### Or try a sample image")
                gr.Examples(examples=[[s] for s in samples], inputs=inp)

        with gr.Column():
            out = gr.Image(type="numpy", format="png",
                           label="Redacted output (downloads as PNG)")
            status = gr.Textbox(label="Status", interactive=False)

    btn.click(process, inputs=[inp, main_only, debug],
              outputs=[out, status])

    gr.Markdown(
        "---\n"
        "*Approach: the YuNet deep-learning face detector (OpenCV DNN) "
        "locates each face and returns eye and mouth landmark points. "
        "Redaction boxes are placed directly on those landmarks, so they "
        "track the real features rather than being estimated from face "
        "proportions. Boxes are semi-translucent to preserve image "
        "context for downstream training.*"
    )

if __name__ == "__main__":
    demo.launch()
