"""
Face redaction engine for LH2 Data Labs.

Detects faces in an image and overlays semi-translucent black rectangles
over the eyes and mouth -- removing identity (PII) while keeping the rest
of the image usable for downstream AI training.

Approach
--------
Detection uses **YuNet**, a compact deep-learning face detector that ships
as a ~230 KB ONNX model and runs through OpenCV's DNN module. YuNet was
chosen over classic Haar cascades because Haar produces many false
detections on busy backgrounds (foliage, textured clothing) and copes
poorly with hair partially covering the face. YuNet is far more robust on
real-world photos.

Crucially, YuNet returns **5 facial landmarks per face** -- right eye,
left eye, nose tip, right mouth corner, left mouth corner -- so the eye
and mouth redaction boxes are placed on the *actual* detected feature
positions rather than estimated from face-box proportions.

The model file is located at runtime by `get_model_path()`, which checks
a few common locations and, if the model is absent, downloads it once
from a public mirror and caches it. This keeps the repo light while still
working with no manual setup.
"""

import os
import cv2
import numpy as np

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


def get_model_path():
    """Return a local path to the YuNet ONNX model.

    The model file is expected to be uploaded into the repo alongside this
    file (or in a 'models/' subfolder). It is small (~230 KB). Shipping it
    in the repo means the tool needs no network access at runtime and
    starts instantly.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, MODEL_FILENAME),
        os.path.join(here, "models", MODEL_FILENAME),
        MODEL_FILENAME,
    ]
    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 50_000:
            return path
    raise RuntimeError(
        f"YuNet model '{MODEL_FILENAME}' not found. Place it next to "
        f"redactor.py (or in a 'models/' subfolder).")


# A single detector instance is created lazily and reused.
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        model_path = get_model_path()
        # input_size is updated per-image before each detect() call.
        _detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,   # confidence cutoff for a real face
            nms_threshold=0.3,
            top_k=5000,
        )
    return _detector


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def _clamp_box(x1, y1, x2, y2, w, h):
    """Order and clamp a box to image bounds; return ints."""
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (
        int(max(0, min(x1, w))), int(max(0, min(y1, h))),
        int(max(0, min(x2, w))), int(max(0, min(y2, h))),
    )


def _draw_box(img, box, opacity=0.80):
    """Overlay a semi-translucent black rectangle over the given region.

    opacity = 1.0 -> solid black; 0.0 -> invisible.
    0.80 destroys identity while staying clearly a redaction.
    """
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, opacity, img, 1.0 - opacity, 0, dst=img)


def _rotated_rect_points(center, size, angle_deg):
    """Return the 4 corner points of a rotated rectangle as an int array.

    center    : (cx, cy)
    size      : (width, height)
    angle_deg : rotation in degrees, clockwise-positive (image coords).
    """
    box = cv2.boxPoints(((float(center[0]), float(center[1])),
                         (float(size[0]), float(size[1])),
                         float(angle_deg)))
    return box.astype(np.int32)


def _draw_rotated_box(img, center, size, angle_deg, opacity=0.80):
    """Overlay a semi-translucent black rotated rectangle.

    Used so the redaction bar can tilt to match a tilted head, keeping the
    eyes / mouth fully covered without an oversized axis-aligned box.
    """
    if size[0] <= 1 or size[1] <= 1:
        return
    pts = _rotated_rect_points(center, size, angle_deg)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (0, 0, 0))
    cv2.addWeighted(overlay, opacity, img, 1.0 - opacity, 0, dst=img)


def _eye_angle_deg(face):
    """Head-tilt angle in degrees, measured from the two eye landmarks.

    0 = eyes level. Positive = the face is rotated so the person's left
    eye is lower (clockwise in image coordinates). The same angle is used
    for both the eye bar and the mouth bar.
    """
    rex, rey = face["right_eye"]
    lex, ley = face["left_eye"]
    return float(np.degrees(np.arctan2(ley - rey, lex - rex)))


def _detect(image_bgr):
    """Run YuNet on an image.

    Returns a list of dicts, one per detected face, each with:
        'box'        : (x, y, w, h) face bounding box
        'right_eye'  : (x, y)
        'left_eye'   : (x, y)
        'nose'       : (x, y)
        'mouth_right': (x, y)
        'mouth_left' : (x, y)
        'score'      : detection confidence
    """
    h, w = image_bgr.shape[:2]
    det = _get_detector()
    det.setInputSize((w, h))
    _, faces = det.detect(image_bgr)

    results = []
    if faces is None:
        return results

    for f in faces:
        # YuNet row layout: [x, y, w, h,
        #   reye_x, reye_y, leye_x, leye_y, nose_x, nose_y,
        #   mr_x, mr_y, ml_x, ml_y, score]
        x, y, bw, bh = f[0], f[1], f[2], f[3]
        results.append({
            "box": (x, y, bw, bh),
            "right_eye": (f[4], f[5]),
            "left_eye": (f[6], f[7]),
            "nose": (f[8], f[9]),
            "mouth_right": (f[10], f[11]),
            "mouth_left": (f[12], f[13]),
            "score": float(f[14]),
        })
    return results


def _landmarks_reliable(face):
    """Return True if a face's eye/mouth landmarks look trustworthy.

    YuNet still fires its face detector on extreme head tilts / profiles,
    but the 5 landmark points it returns for such faces are often
    degenerate (collapsed onto one point, at the origin, or outside the
    face box). When that happens we should fall back to face-box bands
    rather than draw a wrong or empty redaction.
    """
    bx, by, bw, bh = face["box"]
    if bw <= 1 or bh <= 1:
        return False

    pts = [face["right_eye"], face["left_eye"], face["nose"],
           face["mouth_right"], face["mouth_left"]]

    # All landmarks must sit inside (a slightly padded) face box.
    pad_x, pad_y = bw * 0.25, bh * 0.25
    for (px, py) in pts:
        if not (bx - pad_x <= px <= bx + bw + pad_x and
                by - pad_y <= py <= by + bh + pad_y):
            return False

    # Eyes must be separated by a sensible fraction of the face width.
    eye_dx = abs(face["left_eye"][0] - face["right_eye"][0])
    if eye_dx < bw * 0.15:
        return False

    # Mouth corners must be separated too (not collapsed to a point).
    mouth_dx = abs(face["mouth_left"][0] - face["mouth_right"][0])
    if mouth_dx < bw * 0.08:
        return False

    return True


def _eye_box_from_facebox(face, w, h):
    """Fallback eye region derived from the face bounding box.

    Used when landmarks are unreliable (e.g. strongly tilted heads). Less
    precise than the landmark version, but guarantees the eye region is
    still covered so no face is left un-redacted. Returns a rotated-rect
    spec with angle 0 (no reliable landmarks to derive a tilt from).
    """
    bx, by, bw, bh = face["box"]
    # Eyes sit roughly 24-52% down a YuNet face box.
    cx = bx + 0.50 * bw
    cy = by + 0.38 * bh
    return {"center": (cx, cy), "size": (0.92 * bw, 0.28 * bh),
            "angle": 0.0}


def _mouth_box_from_facebox(face, w, h):
    """Fallback mouth region derived from the face bounding box.

    Used when landmarks are unreliable. Covers the lower-centre of the
    face box where the mouth sits. Returns a rotated-rect spec, angle 0.
    """
    bx, by, bw, bh = face["box"]
    # Mouth sits roughly 66-92% down a YuNet face box.
    cx = bx + 0.50 * bw
    cy = by + 0.79 * bh
    return {"center": (cx, cy), "size": (0.64 * bw, 0.26 * bh),
            "angle": 0.0}


def _eye_box_from_landmarks(face, w, h):
    """Eye redaction region built from the two eye landmarks.

    Returns a rotated-rectangle spec {'center','size','angle'}. The bar is
    rotated to match the head tilt (angle of the eye-to-eye line) and its
    length is measured along that tilted line, so it covers the eyes
    cleanly even on a tilted head. A minimum height tied to face size
    keeps the eyes covered if YuNet reports unusually close-set eyes.
    """
    rex, rey = face["right_eye"]
    lex, ley = face["left_eye"]
    _, _, bw, bh = face["box"]
    # True eye separation measured along the tilted eye-line.
    eye_dist = float(np.hypot(lex - rex, ley - rey))
    cx = (rex + lex) / 2.0
    cy = (rey + ley) / 2.0
    angle = _eye_angle_deg(face)
    # Length: span the eyes plus padding for eye corners / brow ends.
    bar_len = eye_dist * 1.84
    # Thickness: slim bar, but never below a safe floor.
    bar_thick = max(eye_dist * 0.56, bh * 0.14)
    return {"center": (cx, cy), "size": (bar_len, bar_thick),
            "angle": angle}


def _mouth_box_from_landmarks(face, w, h):
    """Mouth redaction region built from the two mouth-corner landmarks.

    Returns a rotated-rectangle spec {'center','size','angle'}. Kept as a
    thin bar so it covers the lips without consuming useful surrounding
    pixels (chin, nasolabial folds). The bar is rotated to match the head
    tilt. The mouth and lips must always be covered: a minimum thickness
    tied to face size guards against the bar collapsing too thin.
    """
    mrx, mry = face["mouth_right"]
    mlx, mly = face["mouth_left"]
    _, _, bw, bh = face["box"]
    # True mouth width measured along the tilted mouth-corner line.
    mouth_w = float(np.hypot(mlx - mrx, mly - mry))
    cx = (mrx + mlx) / 2.0
    cy = (mry + mly) / 2.0
    # Tilt the mouth bar by the same head-tilt angle as the eyes.
    angle = _eye_angle_deg(face)
    # Nudge centre downward (perpendicular to the bar) so it sits on the
    # mouth, not the nose. Down-the-face direction is the bar normal.
    a = np.radians(angle)
    nudge = mouth_w * 0.08
    cx += -np.sin(a) * nudge
    cy += np.cos(a) * nudge
    # Length: spans the lip corners (kept wide -- a long, thin bar).
    bar_len = mouth_w * 1.64
    # Thickness: thin, but never below a safe floor so lips stay covered.
    bar_thick = max(mouth_w * 0.52, bh * 0.14)
    return {"center": (cx, cy), "size": (bar_len, bar_thick),
            "angle": angle}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def detect_faces(image_bgr, main_only=True):
    """Detect faces and compute their eye/mouth redaction boxes.

    Args:
        image_bgr: BGR numpy image.
        main_only: if True, keep only the largest face (most reliable for
                   single-subject portraits). If False, keep all faces.

    Returns:
        list of dicts, each with 'face_box', 'eye_box', 'mouth_box',
        'score'.
    """
    h, w = image_bgr.shape[:2]
    faces = _detect(image_bgr)

    if main_only and len(faces) > 1:
        # Keep the largest face by bounding-box area.
        faces = [max(faces, key=lambda f: f["box"][2] * f["box"][3])]

    out = []
    for f in faces:
        bx, by, bw, bh = f["box"]
        if _landmarks_reliable(f):
            eye_box = _eye_box_from_landmarks(f, w, h)
            mouth_box = _mouth_box_from_landmarks(f, w, h)
            method = "landmarks"
        else:
            # Tilted / profile face with unreliable landmarks: fall back
            # to face-box bands so the face is still redacted.
            eye_box = _eye_box_from_facebox(f, w, h)
            mouth_box = _mouth_box_from_facebox(f, w, h)
            method = "facebox-fallback"
        out.append({
            "face_box": _clamp_box(bx, by, bx + bw, by + bh, w, h),
            "eye_box": eye_box,
            "mouth_box": mouth_box,
            "score": f["score"],
            "method": method,
        })
    return out


# Fixed redaction opacity. The box is almost fully opaque so identity is
# genuinely removed (no eye/mouth detail visible), with only the faintest
# translucency so it qualifies as "semi-translucent" per the spec. This is
# deliberately NOT user-adjustable: a privacy tool must not let the user
# weaken the redaction to a point where the face is still recognisable.
REDACTION_OPACITY = 0.92


def redact_image(image_bgr, main_only=True):
    """Redact eyes and mouth on detected faces.

    Args:
        image_bgr: BGR numpy image.
        main_only: redact only the largest face if True, else all faces.

    Returns:
        (redacted_image_bgr, num_faces_redacted)
    """
    img = image_bgr.copy()
    faces = detect_faces(img, main_only=main_only)
    for f in faces:
        for region in (f["eye_box"], f["mouth_box"]):
            _draw_rotated_box(img, region["center"], region["size"],
                              region["angle"], REDACTION_OPACITY)
    return img, len(faces)
