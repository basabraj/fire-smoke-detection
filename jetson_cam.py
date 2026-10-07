import os, sys,time,signal,sqlite3,threading
from datetime import datetime, timezone
from queue import Queue

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import cv2
from flask import Flask, Response
import torch
from ultralytics import YOLO

Gst.init(None)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard"))
from sms_alert import send_fire_alert

PORT = 4207

RTSP_URL = "rtsp://admin:Digital%40123@192.168.96.83:554/live1.sdp"

# ── Dashboard logging (same schema/behavior as rtsp_stream.py) ───────────

LOCATION_NAME = "Home"
CAMERA_NAME = "RTSP-Cam-1"
DASHBOARD_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "dashboard.db")
FOOTAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "static", "footage")
os.makedirs(FOOTAGE_DIR, exist_ok=True)

LOG_COOLDOWN_SECONDS = 15
_last_logged = {}

ALERT_SMS_COOLDOWN_SECONDS = 300  # 5 minutes
_last_alert_sent = {}


def maybe_send_fire_alert(conn, detection_id):
    now = time.time()
    if now - _last_alert_sent.get("fire", 0) < ALERT_SMS_COOLDOWN_SECONDS:
        return
    row = conn.execute("SELECT phone_number FROM alert_recipients ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return
    phone_number = row[0]
    message, status = send_fire_alert(phone_number, LOCATION_NAME, CAMERA_NAME)
    conn.execute(
        "INSERT INTO alert_events (detection_id, phone_number, message, status, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (detection_id, phone_number, message, status,
         datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')),
    )
    conn.commit()
    _last_alert_sent["fire"] = now


def log_detection_to_dashboard(cls_name, conf, frame):
    now = time.time()
    if now - _last_logged.get(cls_name, 0) < LOG_COOLDOWN_SECONDS:
        return
    _last_logged[cls_name] = now

    ts = datetime.now(timezone.utc)
    footage_filename = f"{ts.strftime('%Y%m%d_%H%M%S')}_{cls_name}.jpg"
    cv2.imwrite(os.path.join(FOOTAGE_DIR, footage_filename), frame)

    try:
        conn = sqlite3.connect(DASHBOARD_DB, timeout=5)
        cur = conn.execute(
            "INSERT INTO detections (timestamp, location, camera_name, detection_type, "
            "confidence, is_violation, footage_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts.strftime('%Y-%m-%d %H:%M:%S'), LOCATION_NAME, CAMERA_NAME, cls_name,
             conf, True, footage_filename),
        )
        detection_id = cur.lastrowid
        conn.commit()

        if cls_name == "fire":
            maybe_send_fire_alert(conn, detection_id)

        conn.close()

    except sqlite3.OperationalError as e:
        print(f"[dashboard-log] skipped: {e}")


# ── Model ──────────────────────────────────────────────────────────────

PROCESS_WIDTH = 640

FIRE_CONF = 0.65
SMOKE_CONF = 0.60
MODEL_CONF_FLOOR = min(FIRE_CONF, SMOKE_CONF)

DEBUG_LOG_ALL_DETECTIONS = True
INFER_CONF = 0.30 if DEBUG_LOG_ALL_DETECTIONS else MODEL_CONF_FLOOR

CLASS_MAP = {0: 'fire', 1: 'smoke'}
CLASS_CONF = {0: FIRE_CONF, 1: SMOKE_CONF}
COLORS = {'fire': (0, 0, 139), 'smoke': (139, 0, 0)}  # BGR

model = YOLO('yolo_smoke_fire.pt')
model.to(DEVICE)
print(f"[MODEL] running on {DEVICE}")
app = Flask(__name__)

frame_queue = Queue(maxsize=1)

# ── GStreamer hardware-decode pipeline ────────────────────────────────────

PIPELINE_DESC = (
    f'rtspsrc location="{RTSP_URL}" latency=200 protocols=tcp tcp-timeout=10000000 ! '
    'application/x-rtp,media=video ! decodebin ! '
    'nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! '
    'video/x-raw,format=BGR ! '
    'appsink name=sink max-buffers=1 drop=true sync=false'
)

STALL_TIMEOUT = 10  # seconds with no frames before the watchdog forces a reconnect

last_frame_time = time.time()
pipeline_ref = {'pipeline': None}
stop_event = threading.Event()

latest_ref = {'frame': None, 'seq': 0}
latest_lock = threading.Lock()


def build_pipeline():
    try:
        pipeline = Gst.parse_launch(PIPELINE_DESC)
    except GLib.Error as e:
        raise SystemExit(
            f"[FATAL] Could not build pipeline — check that nvvideo4linux2 "
            f"and decodebin plugins are installed: {e}"
        )

    sink = pipeline.get_by_name('sink')
    if sink is None:
        raise SystemExit("[FATAL] appsink 'sink' element not found in pipeline")

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        raise SystemExit("[FATAL] Pipeline failed to reach PLAYING state — "
                          "check the RTSP URL and camera connectivity")

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


def _no_signals(target):
    """Blocks SIGINT/SIGTERM in this thread so Ctrl+C is handled by the main
    thread instead of landing mid blocking-read here (same reasoning as
    rtsp_stream.py's _no_signals, applied to the GStreamer pull-sample call
    instead of cv2.VideoCapture.read())."""
    def wrapper(*args, **kwargs):
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        target(*args, **kwargs)
    return wrapper


# ── Capture thread — pulls decoded frames off the GPU-decoded appsink ────

def capture_loop():
    global last_frame_time
    pipeline, sink = build_pipeline()

    while not stop_event.is_set():
        sample = sink.emit('pull-sample')
        if sample is None:
            print("[WARN] Pipeline produced no sample, reconnecting...")
            pipeline.set_state(Gst.State.NULL)
            time.sleep(2)
            if stop_event.is_set():
                break
            pipeline, sink = build_pipeline()
            continue

        frame = sample_to_ndarray(sample)
        if frame is None:
            continue

        last_frame_time = time.time()
        with latest_lock:
            latest_ref['frame'] = frame
            latest_ref['seq'] += 1

    pipeline.set_state(Gst.State.NULL)


def watch_for_stall():
    global last_frame_time
    while not stop_event.is_set():
        time.sleep(3)
        if time.time() - last_frame_time > STALL_TIMEOUT:
            print("[WATCHDOG] No frames for a while — feed looks frozen, forcing reconnect...")
            pipeline = pipeline_ref.get('pipeline')
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)
            last_frame_time = time.time()  # avoid re-triggering while it reconnects


def bus_watcher():
    while not stop_event.is_set():
        pipeline = pipeline_ref.get('pipeline')
        if pipeline is None:
            time.sleep(0.2)
            continue
        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(
            200 * Gst.MSECOND,
            Gst.MessageType.ERROR | Gst.MessageType.EOS
        )
        if msg is None:
            continue
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"[ERROR] {err}\n[DEBUG] {debug}")
        elif msg.type == Gst.MessageType.EOS:
            print("[EOS] Stream ended")


# ── Inference thread — always works on the newest available frame ────────

def infer_loop():
    last_seen_seq = -1

    while not stop_event.is_set():
        with latest_lock:
            seq = latest_ref['seq']
            frame = latest_ref['frame']
        if frame is None or seq == last_seen_seq:
            time.sleep(0.005)
            continue
        last_seen_seq = seq
        frame = frame.copy()

        h, w = frame.shape[:2]
        if w > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / w
            frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))

        results = model(frame, conf=INFER_CONF, verbose=False)[0]
        for det in results.boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            cls_name = CLASS_MAP.get(cls_id, f"class{cls_id}")
            required = CLASS_CONF.get(cls_id, MODEL_CONF_FLOOR)
            if conf < required:
                if DEBUG_LOG_ALL_DETECTIONS:
                    print(f"[DEBUG below-threshold] {cls_name} conf={conf:.4f} "
                          f"(needs >= {required:.2f})")
                continue
            print(f"[DETECT] {cls_name} conf={conf:.4f}")
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            color = COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            log_detection_to_dashboard(cls_name, conf, frame)

        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()
                frame_queue.put(frame)
            except Exception:
                pass


# ── Frame streaming to HTML ────────────────────────────────────────────

def gen_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ok, buf = cv2.imencode('.jpg', frame)
            if not ok:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        else:
            time.sleep(0.01)


@app.route('/')
def index():
    return (
        '<html><body style="margin:0;background:#111;display:flex;'
        'justify-content:center;align-items:center;min-height:100vh">'
        '<img src="/video_feed" style="max-width:100%">'
        '</body></html>'
    )


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    threading.Thread(target=_no_signals(capture_loop), daemon=True).start()
    threading.Thread(target=_no_signals(infer_loop), daemon=True).start()
    threading.Thread(target=_no_signals(watch_for_stall), daemon=True).start()
    threading.Thread(target=_no_signals(bus_watcher), daemon=True).start()

    print(f"Jetson hardware-decode + GPU inference running. "
          f"Open http://localhost:{PORT}/ — Ctrl+C to quit.")
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False)
    finally:
        stop_event.set()
        pipeline = pipeline_ref.get('pipeline')
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
