import os
from dotenv import load_dotenv
load_dotenv()

import time
import threading
from queue import Queue

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import numpy as np
import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

from detection_logger import log_detection, FrameArchiver, TemporalConfirmer

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only
Gst.init(None)

# ── CONFIG ────────────────────────────────────────────────────────────────
# No hardcoded fallback on purpose — a camera password baked into source is
# a leak waiting to happen the moment this repo is pushed anywhere. Fails
# loudly instead of silently running with a stale default if .env is missing.
RTSP_URL = os.environ.get('RTSP_URL')
if not RTSP_URL:
    raise SystemExit(
        "RTSP_URL is not set. Copy .env.example to .env and fill in your "
        "camera's real RTSP URL."
    )
PORT = int(os.environ.get('PORT', 4000))
CAMERA_NAME = os.environ.get('CAMERA_NAME', 'Tiandy-Office-CBSIOT-4')

# If no frame arrives for this long, assume the pipeline is stuck and force a reconnect
STALL_TIMEOUT = 10  # seconds

# YOLO's own default is 0.25 if you don't pass conf= — way too permissive for a
# fixed indoor camera (random background clutter starts reading as "Smoke").
# Per-class floor for drawing/logging a box at all: Smoke's set higher than
# Fire's because it's the one that's shown false positives on this camera
# (dark hair/skin texture in the office scene misread as "Smoke").
FIRE_CONF = 0.6
SMOKE_CONF = 0.65

# We still need one number to pass into the model() call itself — use the
# lower of the two so candidate boxes for BOTH classes come through, then
# filter each detection against its own class's threshold below.
MODEL_CONF_FLOOR = min(FIRE_CONF, SMOKE_CONF)

# ALERT_CONF: stricter bar for actually firing an alert, so a borderline box
# on screen doesn't spam alerts. Currently only wired up for Fire.
ALERT_CONF = 0.7

# NOTE: this checkpoint's embedded model.names metadata says {0: 'Fire',
# 1: 'Smoke'} — but that metadata is WRONG for what the model actually
# learned. Tested directly against unambiguous reference photos (clear
# flame-only fire, clear industrial fire) and in both cases the network's
# class-1 predictions land squarely on the flames — i.e. the model was
# really trained with Smoke=0/Fire=1, and only the saved .pt names dict is
# mislabeled (a known way this can happen: names list order not matching
# the training data.yaml). Trust the empirical mapping below, not
# model.names.
CLASS_MAP = {0: 'Smoke', 1: 'Fire'}
# Keyed by the same class id as CLASS_MAP — keep the two in sync.
CLASS_CONF = {0: SMOKE_CONF, 1: FIRE_CONF}
COLORS = {'Fire': (0, 0, 139), 'Smoke': (139, 0, 0)}  # BGR: Fire=deep red, Smoke=deep blue

PIPELINE_DESC = (
    f'rtspsrc location="{RTSP_URL}" latency=200 protocols=tcp tcp-timeout=10000000 ! '
    'rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! '
    'video/x-raw,format=BGR ! '
    'appsink name=sink max-buffers=1 drop=true sync=false'
)

model = YOLO('weights/best.pt')
model.to('cpu')
app = Flask(__name__)

frame_queue = Queue(maxsize=1)
last_frame_time = time.time()
pipeline_ref = {'pipeline': None}
archiver = FrameArchiver(positive_interval=5)
confirmer = TemporalConfirmer(required_hits=3, window=5, iou_threshold=0.3)


def build_pipeline():
    pipeline = Gst.parse_launch(PIPELINE_DESC)
    sink = pipeline.get_by_name('sink')
    pipeline.set_state(Gst.State.PLAYING)
    pipeline_ref['pipeline'] = pipeline
    return pipeline, sink


def sample_to_ndarray(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value('width')
    height = structure.get_value('height')

    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
        frame = frame.reshape((height, width, 3)).copy()  # already BGR
    finally:
        buf.unmap(mapinfo)
    return frame


# ── Watchdog Thread ────────────────────────────────────
# appsink's pull-sample() can block indefinitely on a stalled RTSP
# connection (same failure mode as OpenCV's cap.read()). This thread
# force-kills the pipeline from the outside if no frame has landed in a
# while, which unblocks the stuck pull-sample() call in detect_loop so it
# can reconnect instead of freezing the feed.
def watch_for_stall():
    global last_frame_time
    while True:
        time.sleep(3)
        if time.time() - last_frame_time > STALL_TIMEOUT:
            print("⚠️ No frames for a while — feed looks frozen, forcing reconnect...")
            pipeline = pipeline_ref.get('pipeline')
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)
            last_frame_time = time.time()  # avoid re-triggering while it reconnects


# ── Background Detection Thread ───────────────────────
def detect_loop():
    global last_frame_time
    pipeline, sink = build_pipeline()
    last_alert_time = 0

    while True:
        sample = sink.emit('pull-sample')
        if sample is None:
            # EOS, error, or force-killed by the watchdog — reconnect
            print("⚠️ Pipeline produced no sample, reconnecting...")
            pipeline.set_state(Gst.State.NULL)
            time.sleep(2)
            pipeline, sink = build_pipeline()
            continue

        frame = sample_to_ndarray(sample)
        if frame is None:
            continue

        last_frame_time = time.time()
        clean_frame = frame.copy()  # pre-annotation, for archiving/retraining

        results = model(frame, conf=MODEL_CONF_FLOOR)[0]
        detections = [
            (int(det.cls[0]), float(det.conf[0]), *map(int, det.xyxy[0]))
            for det in results.boxes
            if float(det.conf[0]) >= CLASS_CONF.get(int(det.cls[0]), MODEL_CONF_FLOOR)
        ]  # (cls_id, conf, x1, y1, x2, y2)

        # Require the detection to persist across recent frames before it's
        # treated as real — filters one-off single-frame misreads (see
        # TemporalConfirmer docstring) without touching the confidence bar.
        confirmed = confirmer.confirm(detections)

        for cls_id, conf, x1, y1, x2, y2, track_id in confirmed:
            cls_name = CLASS_MAP.get(cls_id, f"Class {cls_id}")

            if cls_name == "Fire" and conf > ALERT_CONF and time.time() - last_alert_time > 10:
                print("🔥 FIRE DETECTED")  # Replace with your alert function
                last_alert_time = time.time()

            color = COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Metadata log (every confirmed detection, regardless of frame-save
        # throttling) and throttled frame archiving — only when fire/smoke
        # is actually confirmed.
        if confirmed:
            frame_path = archiver.maybe_save_positive(
                clean_frame, [(c, x1, y1, x2, y2) for c, _, x1, y1, x2, y2, _ in confirmed],
                annotated_frame=frame
            )
            for cls_id, conf, x1, y1, x2, y2, track_id in confirmed:
                cls_name = CLASS_MAP.get(cls_id, f"Class {cls_id}")
                log_detection(CAMERA_NAME, cls_name, conf, (x1, y1, x2, y2), frame_path)

        # Keep only the latest frame
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()
                frame_queue.put(frame)
            except Exception:
                pass


# ── Frame Streaming to HTML ──────────────────────────
def gen_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.01)


@app.route('/')
def index():
    return render_template('webcam.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Entry Point ──────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=detect_loop, daemon=True).start()
    threading.Thread(target=watch_for_stall, daemon=True).start()
    app.run(host='0.0.0.0', port=PORT, debug=False)
