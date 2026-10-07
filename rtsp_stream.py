import os, cv2

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000|max_delay;500000'

import sys
import time
import signal
import sqlite3
import threading
from datetime import datetime, timezone
from queue import Queue

from flask import Flask, Response
import torch
from ultralytics import YOLO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard"))
from sms_alert import send_fire_alert

PORT = 4206

RTSP_URL = "rtsp://admin:Digital@123@192.168.96.83:554/live1.sdp"

# ── Dashboard logging ─────────────────────────────────────────────────────

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
    """Looks up the Alert page's saved recipient and fires the emergency SMS
    (see dashboard/sms_alert.py) — only for 'fire', and only if the cooldown
    has elapsed. Runs on the same sqlite3 connection as the detection insert
    so this is one round-trip, not two."""
    now = time.time()
    if now - _last_alert_sent.get("fire", 0) < ALERT_SMS_COOLDOWN_SECONDS:
        return
    row = conn.execute("SELECT phone_number FROM alert_recipients ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return  # no recipient saved on the Alert page yet
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

STALL_TIMEOUT = 10


PROCESS_WIDTH = 640

FIRE_CONF = 0.65
SMOKE_CONF = 0.60
MODEL_CONF_FLOOR = min(FIRE_CONF, SMOKE_CONF)


DEBUG_LOG_ALL_DETECTIONS = True
INFER_CONF = 0.30 if DEBUG_LOG_ALL_DETECTIONS else MODEL_CONF_FLOOR

CLASS_MAP = {0: 'fire', 1: 'smoke'}  # empirically verified, see Home/picam_stream.py
CLASS_CONF = {0: FIRE_CONF, 1: SMOKE_CONF}
COLORS = {'fire': (0, 0, 139), 'smoke': (139, 0, 0)}  # BGR: fire=deep red, smoke=deep blue

model = YOLO('yolo_smoke_fire.pt')
model.to(DEVICE)
app = Flask(__name__)

frame_queue = Queue(maxsize=1)
last_frame_time = time.time()


latest_ref = {'frame': None, 'seq': 0}
latest_lock = threading.Lock()


def open_capture():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    return cap


# ── Watchdog Thread ────────────────────────────────────

def watch_for_stall():
    while True:
        time.sleep(3)
        if time.time() - last_frame_time > STALL_TIMEOUT:
            print("[WATCHDOG] No frames for a while — feed looks frozen "
                  "(waiting for the ffmpeg-level socket timeout to reconnect on its own)...")


# ── Capture Thread — reads the camera and NOTHING else ───────────────────

def capture_loop():
    global last_frame_time
    cap = open_capture()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Stream read failed, reconnecting...")
            cap.release()
            time.sleep(2)
            cap = open_capture()
            continue

        last_frame_time = time.time()
        with latest_lock:
            latest_ref['frame'] = frame
            latest_ref['seq'] += 1


# ── Inference Thread — always works on the newest available frame ────────
def infer_loop():
    last_seen_seq = -1

    while True:
        with latest_lock:
            seq = latest_ref['seq']
            frame = latest_ref['frame']
        if frame is None or seq == last_seen_seq:
            time.sleep(0.005)  # nothing new yet — don't busy-spin
            continue
        last_seen_seq = seq
        frame = frame.copy()  

        # Downscale before inference/display (see PROCESS_WIDTH note above).
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
                continue  # below this class's own floor — skip drawing/alerting
            print(f"[DETECT] {cls_name} conf={conf:.4f}")
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            color = COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            log_detection_to_dashboard(cls_name, conf, frame)

        # Keep only the latest processed frame for the HTTP stream
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


def _no_signals(target):
    """See camera_worker.py's _no_signals for why: without this, Ctrl+C can
    land in the capture thread mid-ffmpeg-network-read and abort the whole
    process instead of raising a normal KeyboardInterrupt in main."""
    def wrapper(*args, **kwargs):
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        target(*args, **kwargs)
    return wrapper


if __name__ == '__main__':
    threading.Thread(target=_no_signals(capture_loop), daemon=True).start()
    threading.Thread(target=_no_signals(infer_loop), daemon=True).start()
    threading.Thread(target=_no_signals(watch_for_stall), daemon=True).start()
    print(f"RTSP + Home model running. Open http://localhost:{PORT}/ — Ctrl+C to quit.")
    app.run(host='0.0.0.0', port=PORT, debug=False)

