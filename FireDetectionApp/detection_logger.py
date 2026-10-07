"""
Shared persistence helpers for the CCTV detection streams
(cctv_stream.py / cctv_stream_gst.py):

1. SQLite metadata log — one row per detection: timestamp, camera, class,
   confidence, box coordinates. This is the "Record timestamp and location
   metadata" step from the reference pipeline.

2. FrameArchiver — throttled saving of raw (unannotated) frames into
   collected_data/images + collected_data/labels in YOLO label format
   whenever fire/smoke is actually detected, so they can be dropped
   straight into a future retraining run. This is the "Store frame in
   database for training" branch from the reference pipeline.

3. Same throttling also saves the annotated frame (boxes + labels drawn)
   into outputs/, for quick human review of what actually got flagged —
   e.g. spotting a recurring false positive without having to redraw boxes
   from the DB by hand.

4. TemporalConfirmer — requires a detection to reappear in roughly the same
   spot across several recent frames before it counts as "confirmed" (i.e.
   worth drawing/logging/archiving at all). This catches one-off single-
   frame misreads — a person's hair or a moving arm briefly resembling
   smoke for a single frame — without having to push the raw confidence
   threshold so high that real early/thin smoke would get buried with it.
   It will NOT filter a misread that itself sits still and persists for
   several seconds (e.g. a static chair headrest) — that's a genuinely
   different problem (the model itself needs correcting on that texture),
   not a transient-noise problem.

   On top of that, each confirmed track's box-area history is watched and
   ENFORCED — inspired by the AVT (area-variation tracker) idea that real
   fire/smoke keeps changing area (grows, flickers, drifts) while a static
   false positive (a chair headrest, a red folder) barely changes at all.
   A track only passes once its area's coefficient of variation (std/mean
   of recent areas) reaches `area_cv_thresh`; a track with too little
   history yet (fewer than 2 area samples) counts as CV=0 and is held back
   too, same as the reference AVT behaviour for a brand-new object.
   Deliberately does NOT adopt the reference AVT implementation's "trust a
   track permanently once it varies once" rule — a single noisy jump could
   then never be re-checked — CV is recomputed fresh every time instead.
   This does mean a couple of extra frames of latency beyond the
   persistence gate before anything new gets confirmed; that's the
   trade-off for not needing the confidence threshold pushed high enough
   to bury real early/thin smoke.

Throttled to once every few seconds because a fixed camera streaming at
~1 fps of inference would otherwise dump near-duplicate frames of the same
ongoing event — a training set wants variety, not volume.
"""

import os
import sqlite3
import threading
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'detections.db')
DATA_DIR = os.path.join(BASE_DIR, 'collected_data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')  # annotated frames, for human review

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

_db_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        camera      TEXT NOT NULL,
        class_name  TEXT NOT NULL,
        confidence  REAL NOT NULL,
        x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
        frame_path  TEXT
    )
''')
_conn.commit()


def log_detection(camera, class_name, confidence, box, frame_path=None):
    """Persist one detection row. box = (x1, y1, x2, y2) in pixel coords."""
    x1, y1, x2, y2 = box
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with _db_lock:
        _conn.execute(
            'INSERT INTO detections '
            '(timestamp, camera, class_name, confidence, x1, y1, x2, y2, frame_path) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (ts, camera, class_name, float(confidence), x1, y1, x2, y2, frame_path)
        )
        _conn.commit()


class FrameArchiver:
    """Throttled saver for fire/smoke detection frames.

    maybe_save_positive() writes the clean frame + a YOLO-format .txt label
    (class_id x_center y_center width height, all normalized 0..1) into
    collected_data/, and — if given the annotated version — a matching
    boxes-drawn copy into outputs/ for quick human review. Only called when
    there's an actual detection; frames with nothing detected are not saved.
    """

    def __init__(self, positive_interval=5):
        self.positive_interval = positive_interval
        self._last_positive = 0

    def maybe_save_positive(self, frame, detections, annotated_frame=None):
        """detections: list of (cls_id, x1, y1, x2, y2) in pixel coords.
        annotated_frame: same frame with boxes/labels already drawn on it,
        if you want a human-reviewable copy saved to outputs/ too."""
        now = time.time()
        if now - self._last_positive < self.positive_interval:
            return None
        self._last_positive = now

        stamp = time.strftime('%Y%m%d_%H%M%S')
        name = f'{stamp}_pos'
        img_path = os.path.join(IMAGES_DIR, f'{name}.jpg')
        label_path = os.path.join(LABELS_DIR, f'{name}.txt')

        cv2.imwrite(img_path, frame)
        h, w = frame.shape[:2]
        with open(label_path, 'w') as f:
            for cls_id, x1, y1, x2, y2 in detections:
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f'{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')

        if annotated_frame is not None:
            cv2.imwrite(os.path.join(OUTPUTS_DIR, f'{name}.jpg'), annotated_frame)

        return img_path


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TemporalConfirmer:
    """Requires a detection to reappear in roughly the same location across
    several recent frames before treating it as confirmed. Filters out
    single-frame flukes (a hair flick, a moving arm briefly resembling
    smoke) without raising the confidence threshold — which would also
    bury real, but naturally low-confidence, early/thin smoke.

    Does NOT help against a misread that itself sits still for a long time
    (a static chair headrest keeps "reappearing in the same spot" just
    like real smoke would) — that needs fixing at the model/threshold
    level instead.
    """

    def __init__(self, required_hits=3, window=5, iou_threshold=0.3,
                 area_history_len=10, track_max_age=10, area_cv_thresh=0.05):
        self.required_hits = required_hits
        self.window = window
        self.iou_threshold = iou_threshold
        self._history = []  # list of per-frame detection lists

        # AVT-style area tracking, enforced (see module docstring). Each
        # track: {'cls_id', 'box', 'areas': [...], 'age'}.
        self.area_history_len = area_history_len
        self.track_max_age = track_max_age  # frames of absence before a track is forgotten
        self.area_cv_thresh = area_cv_thresh
        self._tracks = {}
        self._next_track_id = 0

    def _match_track(self, cls_id, box):
        best_id, best_iou = None, self.iou_threshold
        for tid, t in self._tracks.items():
            if t['cls_id'] != cls_id:
                continue
            iou = _iou(box, t['box'])
            if iou >= best_iou:
                best_id, best_iou = tid, iou
        return best_id

    def _update_track_and_log_cv(self, cls_id, box):
        """Match this confirmed detection to a persistent track, append its
        area to that track's history, compute+log the running coefficient
        of variation, and return (track_id, cv). cv is 0.0 when there's
        not yet enough history (fewer than 2 samples) — same as a track
        whose area genuinely hasn't moved."""
        tid = self._match_track(cls_id, box)
        if tid is None:
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = {'cls_id': cls_id, 'box': box, 'areas': [], 'age': 0}

        track = self._tracks[tid]
        track['box'] = box
        track['age'] = 0
        x1, y1, x2, y2 = box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        track['areas'].append(area)
        if len(track['areas']) > self.area_history_len:
            track['areas'].pop(0)

        cv = 0.0
        if len(track['areas']) >= 2:
            arr = np.array(track['areas'], dtype=float)
            mean = arr.mean()
            cv = float(arr.std() / mean) if mean > 0 else 0.0
            print(f"[AVT-CV] track={tid} class={cls_id} area_cv={cv:.4f} "
                  f"n={len(track['areas'])} areas={[int(a) for a in track['areas']]}")

        return tid, cv

    def confirm(self, detections):
        """detections: list of (cls_id, conf, x1, y1, x2, y2) for the
        current frame. Returns the subset that has been seen (same class,
        overlapping box) in at least `required_hits` of the last `window`
        frames, including this one — as (cls_id, conf, x1, y1, x2, y2,
        track_id). track_id is the actual matched persistent track (see
        _update_track_and_log_cv/_match_track) — unlike the reference AVT
        detect.py, which grabs `list(centroids.keys())[idx]` and silently
        assumes frame-detection order lines up with track-registration
        order (it doesn't, once more than one object is being tracked)."""
        self._history.append(detections)
        if len(self._history) > self.window:
            self._history.pop(0)

        confirmed = []
        matched_ids = set()
        for cls_id, conf, x1, y1, x2, y2 in detections:
            hits = 0
            for past_frame in self._history:
                if any(
                    pcls == cls_id and _iou((x1, y1, x2, y2), (px1, py1, px2, py2)) >= self.iou_threshold
                    for pcls, _, px1, py1, px2, py2 in past_frame
                ):
                    hits += 1
            if hits >= self.required_hits:
                # Keep updating the track's area history even while it's
                # being held back by the CV gate below — it needs more
                # samples to ever prove itself, and aging it out here would
                # reset that progress every single frame.
                tid, cv = self._update_track_and_log_cv(cls_id, (x1, y1, x2, y2))
                matched_ids.add(tid)
                if cv >= self.area_cv_thresh:
                    confirmed.append((cls_id, conf, x1, y1, x2, y2, tid))

        # Age out tracks that weren't matched this round; forget them once
        # stale so memory doesn't grow unboundedly on a 24/7 stream (unlike
        # the reference AVT code, which never expires a track at all).
        for tid in list(self._tracks.keys()):
            if tid not in matched_ids:
                self._tracks[tid]['age'] += 1
                if self._tracks[tid]['age'] > self.track_max_age:
                    del self._tracks[tid]

        return confirmed
