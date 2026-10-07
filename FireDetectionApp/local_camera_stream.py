import os
from dotenv import load_dotenv
load_dotenv()

import time
import threading
import queue

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

from detection_logger import log_detection, FrameArchiver, TemporalConfirmer

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only
Gst.init(None)

# ── CONFIG ────────────────────────────────────────────────────────────────
WIDTH, HEIGHT, FPS = 640, 480, 30
PORT = int(os.environ.get('PORT', 4002))  # different from cctv_stream*.py's 4000 so both can run at once
CAMERA_NAME = os.environ.get('CAMERA_NAME', 'Pi-Local-Camera')


STALL_TIMEOUT = 10  # seconds

FIRE_CONF = 0.05  # TEMP: dropped low for lighter-flame confidence-score testing — restore to 0.6 afterward
SMOKE_CONF = 0.65
MODEL_CONF_FLOOR = min(FIRE_CONF, SMOKE_CONF)
ALERT_CONF = 0.7


CLASS_MAP = {0: 'Smoke', 1: 'Fire'}
CLASS_CONF = {0: SMOKE_CONF, 1: FIRE_CONF}  # keyed by the same id as CLASS_MAP
COLORS = {'Fire': (0, 0, 139), 'Smoke': (139, 0, 0)}  # BGR: Fire=deep red, Smoke=deep blue

model = YOLO('weights/best.pt')
model.to('cpu')
app = Flask(__name__)

raw_queue = queue.Queue(maxsize=1)      # latest camera frame, filled by GStreamer callback
display_queue = queue.Queue(maxsize=1)  # latest annotated frame, consumed by Flask
archiver = FrameArchiver(positive_interval=5)
confirmer = TemporalConfirmer(required_hits=3, window=5, iou_threshold=0.3)

last_frame_time = time.time()
pipeline_ref = {'pipeline': None}
reconnect_lock = threading.Lock()


def on_new_sample(appsink, data):
    global last_frame_time
    sample = appsink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR

    buf = sample.get_buffer()
    size = buf.get_size()
    expected_size = WIDTH * HEIGHT * 3
    if size < expected_size:
        # reshape below crash the callback (and take the pipeline with it).
        print(f"[WARN] Unexpected buffer size {size}, expected {expected_size} — dropping frame")
        return Gst.FlowReturn.OK

    frame = np.ndarray(
        (HEIGHT, WIDTH, 3),
        buffer=buf.extract_dup(0, expected_size),
        dtype=np.uint8
    ).copy()

    last_frame_time = time.time()

    if raw_queue.full():
        try:
            raw_queue.get_nowait()
        except queue.Empty:
            pass
    raw_queue.put(frame)

    return Gst.FlowReturn.OK


def on_bus_message(bus, message):
    t = message.type
    if t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[ERROR] {err}\n[DEBUG] {debug}")
        reconnect()
    elif t == Gst.MessageType.EOS:
        print("[EOS] Stream ended — reconnecting...")
        reconnect()



def build_pipeline():
    global last_frame_time
    pipeline = Gst.parse_launch(
        f"libcamerasrc ! "
        f"video/x-raw,format=RGBx,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
    )
    sink = pipeline.get_by_name("sink")
    sink.connect("new-sample", on_new_sample, sink)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message)

    pipeline.set_state(Gst.State.PLAYING)
    pipeline_ref['pipeline'] = pipeline
    last_frame_time = time.time()  # reset so the watchdog gives the new pipeline time to warm up
    return pipeline


# ── Reconnect / Watchdog ──────────────────────────────────
def reconnect():
    if not reconnect_lock.acquire(blocking=False):
        return  
    try:
        old = pipeline_ref.get('pipeline')
        if old is not None:
            old.set_state(Gst.State.NULL)
        time.sleep(1)  # give libcamera a moment to release the camera device
        build_pipeline()
        print("[INFO] Pi camera pipeline reconnected.")
    finally:
        reconnect_lock.release()


def watch_for_stall():
    while True:
        time.sleep(3)
        if time.time() - last_frame_time > STALL_TIMEOUT:
            print("⚠️ No frames from Pi camera for a while — feed looks frozen, forcing reconnect...")
            reconnect()


glib_loop = GLib.MainLoop()


# ── Detection Thread ───────────────────────────────────
def detect_loop():
    last_alert_time = 0
    while True:
        try:
            frame = raw_queue.get(timeout=1.0)
        except queue.Empty:
            continue

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

        # Metadata log (every confirmed detection) and throttled frame
        # archiving — only when fire/smoke is actually confirmed.
        if confirmed:
            frame_path = archiver.maybe_save_positive(
                clean_frame, [(c, x1, y1, x2, y2) for c, _, x1, y1, x2, y2, _ in confirmed],
                annotated_frame=frame
            )
            for cls_id, conf, x1, y1, x2, y2, track_id in confirmed:
                cls_name = CLASS_MAP.get(cls_id, f"Class {cls_id}")
                log_detection(CAMERA_NAME, cls_name, conf, (x1, y1, x2, y2), frame_path)

        if display_queue.full():
            try:
                display_queue.get_nowait()
            except queue.Empty:
                pass
        display_queue.put(frame)


# ── Frame Streaming to HTML ──────────────────────────
def gen_frames():
    while True:
        try:
            frame = display_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('webcam.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Entry Point ──────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=glib_loop.run, daemon=True).start()
    build_pipeline()
    threading.Thread(target=detect_loop, daemon=True).start()
    threading.Thread(target=watch_for_stall, daemon=True).start()
    print(f"Pi camera running. Open http://<pi-ip>:{PORT}/ — Ctrl+C to quit.")
    app.run(host='0.0.0.0', port=PORT, debug=False)
