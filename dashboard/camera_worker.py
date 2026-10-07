"""
camera_worker.py — the live RTSP capture + YOLO inference pipeline, merged
into the dashboard process so a camera URL saved on the Settings page
connects immediately, with no separate script/process to start by hand
(Home/rtsp_stream.py used to run this standalone against a hardcoded URL —
that file still works stand-alone, but don't run it against the same
camera at the same time as this one: two RTSP connections to one camera
caused visible quality/decode problems earlier in this project).

Capture is done through GStreamer with `decodebin`, which autoplugs the
hardware `nvv4l2decoder` (it has a higher plugin rank than any software
decoder on this Jetson) instead of decoding H.264/H.265 on the CPU the way
cv2.VideoCapture(..., cv2.CAP_FFMPEG) used to — that was the main reason
the dashboard's live view lagged even on the Jetson. `decodebin` also means
this doesn't care whether the camera happens to be sending H.264 or H.265.

Same capture-thread / infer-thread split as before, for the same reason:
GStreamer's blocking pull-sample() isn't something you want sharing a
thread with YOLO inference — running both in one thread would throttle the
RTSP socket enough to desync decode. Detections + fire alerts are written
straight through Flask-SQLAlchemy (with app.app_context(), the same
pattern scheduler.py already uses from its own background thread).

Each call to start() captures its own {rtsp_url, location, camera_name,
fire_conf, smoke_conf} snapshot and hands it to the new threads as a plain
argument, never a shared mutable attribute — so an old, still-shutting-down
generation can never attribute a detection to the *new* camera/location
while it's dying.
"""
import os
import time
import signal
import threading
from datetime import datetime, timezone
from urllib.parse import quote

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from sms_alert import send_fire_alert

Gst.init(None)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(BASE_DIR)
FOOTAGE_DIR = os.path.join(BASE_DIR, "static", "footage")
os.makedirs(FOOTAGE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(HOME_DIR, "yolo_smoke_fire.pt")
PROCESS_WIDTH = 640
CLASS_MAP = {0: 'fire', 1: 'smoke'}
COLORS = {'fire': (0, 0, 139), 'smoke': (139, 0, 0)}  # BGR

LOG_COOLDOWN_SECONDS = 15
ALERT_SMS_COOLDOWN_SECONDS = 300  # 5 minutes
STALL_TIMEOUT = 10
MAX_OPEN_FAILURES_BEFORE_ERROR = 5

# `decodebin` picks the codec (H.264/H.265/...) at runtime and hands it to
# whichever decoder GStreamer ranks highest — nvv4l2decoder (hardware) on
# this box, confirmed via gst-inspect-1.0.
PIPELINE_TEMPLATE = (
    'rtspsrc location="{url}" latency=200 protocols=tcp tcp-timeout=10000000 ! '
    'application/x-rtp,media=video ! decodebin ! '
    'nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! '
    'video/x-raw,format=BGR ! '
    'appsink name=sink max-buffers=1 drop=true sync=false'
)

_model = None
_model_lock = threading.Lock()


def _no_signals(target):
    """Wraps a thread target so SIGINT/SIGTERM are blocked *in that thread*
    before it does anything. Without this, the kernel can route Ctrl+C to
    whichever thread happens to be running — if that's the capture thread
    mid blocking pull-sample() call, an interrupted syscall there can raise
    across a C boundary GStreamer's bindings can't safely unwind. Blocking
    the signal here guarantees the kernel delivers it to the main thread
    instead, where Python's normal KeyboardInterrupt handling applies."""
    def wrapper(*args, **kwargs):
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        target(*args, **kwargs)
    return wrapper


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = YOLO(MODEL_PATH)
            _model.to(DEVICE)
    return _model


def _sanitize_rtsp_url(url):
    """Settings-page passwords can contain '@' (this camera's does), which
    collides with the user:pass@host separator in a URI. rtspsrc's parser
    reads that literally and fails auth, so percent-encode the credential
    portion before handing the URL to Gst.parse_launch. Host/IP never
    contains '@', so splitting the authority on the *last* '@' before the
    first '/' always finds the real separator regardless of what's in the
    password."""
    if '://' not in url or '@' not in url:
        return url
    scheme, rest = url.split('://', 1)
    authority, _, tail = rest.partition('/')
    if '@' not in authority:
        return url
    userinfo, host = authority.rsplit('@', 1)
    user, _, passwd = userinfo.partition(':')
    safe_user = quote(user, safe='')
    safe_pass = quote(passwd, safe='')
    creds = f"{safe_user}:{safe_pass}" if passwd else safe_user
    return f"{scheme}://{creds}@{host}" + (f"/{tail}" if tail else "")


def _sample_to_ndarray(sample):
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


class CameraWorker:
    """The dashboard runs a single instance of this. Saving a new config on
    the Settings page calls .start() again, which cleanly signals any
    previous generation to stop before spinning up the new one."""

    def __init__(self):
        self._stop_event = None
        self._threads = []
        self._latest_lock = threading.Lock()
        self._latest_ref = {'frame': None, 'seq': 0}
        self._annotated_lock = threading.Lock()
        self._latest_annotated = None
        self._last_logged = {}
        self._last_alert_sent = {}
        self._status = {"state": "stopped", "rtsp_url": None, "message": None, "last_frame_at": None}
        self._last_frame_time = 0
        self._app = None
        self._gst_pipeline = None

    # ── public API ──────────────────────────────────────────────────────
    def start(self, app, rtsp_url, location, camera_name, fire_conf, smoke_conf):
        self.stop()
        self._app = app
        self._status = {"state": "connecting", "rtsp_url": rtsp_url, "message": None, "last_frame_at": None}
        with self._latest_lock:
            self._latest_ref = {'frame': None, 'seq': 0}
        with self._annotated_lock:
            self._latest_annotated = None
        self._last_frame_time = 0
        self._gst_pipeline = None

        stop_event = threading.Event()
        self._stop_event = stop_event
        cfg = {
            "rtsp_url": rtsp_url, "location": location, "camera_name": camera_name,
            "fire_conf": fire_conf, "smoke_conf": smoke_conf,
        }
        threads = [
            threading.Thread(target=_no_signals(self._capture_loop), args=(stop_event, cfg), daemon=True),
            threading.Thread(target=_no_signals(self._infer_loop), args=(stop_event, cfg), daemon=True),
            threading.Thread(target=_no_signals(self._watch_loop), args=(stop_event,), daemon=True),
        ]
        self._threads = threads
        for t in threads:
            t.start()

    def stop(self):
        """Signals the current generation's threads to stop and waits for
        them to actually exit before returning. Setting the GStreamer
        pipeline to NULL here (in addition to setting stop_event) is what
        actually unblocks a capture thread parked in a blocking
        pull-sample() call — without it, .join() below could wait forever
        on a camera that's gone silent. This matters specifically because
        the threads may be doing CUDA work: if the process exits (e.g.
        Ctrl+C) while one is still mid-inference, CUDA's own shutdown can
        race an abrupt daemon-thread kill and abort the whole process
        natively (uncatchable from Python) — joining here first means
        app.py's shutdown path always waits this out instead of hitting it."""
        if self._stop_event is not None:
            self._stop_event.set()
        pipeline = self._gst_pipeline
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []
        self._status["state"] = "stopped"

    def get_status(self):
        return dict(self._status)

    def get_jpeg(self):
        with self._annotated_lock:
            frame = self._latest_annotated
        if frame is None:
            return None
        ok, buf = cv2.imencode('.jpg', frame)
        return buf.tobytes() if ok else None

    # ── internals ───────────────────────────────────────────────────────
    def _open_gst(self, rtsp_url):
        desc = PIPELINE_TEMPLATE.format(url=_sanitize_rtsp_url(rtsp_url))
        try:
            pipeline = Gst.parse_launch(desc)
        except GLib.Error as e:
            print(f"[camera_worker] failed to build GStreamer pipeline: {e}")
            return None, None

        sink = pipeline.get_by_name('sink')
        if sink is None:
            print("[camera_worker] appsink 'sink' not found in pipeline")
            return None, None

        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            print("[camera_worker] pipeline failed to reach PLAYING state")
            pipeline.set_state(Gst.State.NULL)
            return None, None

        self._gst_pipeline = pipeline
        return pipeline, sink

    def _capture_loop(self, stop_event, cfg):
        pipeline, sink = self._open_gst(cfg["rtsp_url"])
        open_failures = 0

        while not stop_event.is_set():
            if pipeline is None:
                open_failures += 1
                state = "error" if open_failures >= MAX_OPEN_FAILURES_BEFORE_ERROR else "connecting"
                self._status.update(state=state, rtsp_url=cfg["rtsp_url"],
                                     message="pipeline failed to start — check the URL/credentials, retrying"
                                     if state == "error" else "reconnecting")
                time.sleep(2)
                if stop_event.is_set():
                    break
                pipeline, sink = self._open_gst(cfg["rtsp_url"])
                continue

            sample = sink.emit('pull-sample')
            if stop_event.is_set():
                break
            if sample is None:
                open_failures += 1
                state = "error" if open_failures >= MAX_OPEN_FAILURES_BEFORE_ERROR else "connecting"
                self._status.update(state=state, rtsp_url=cfg["rtsp_url"],
                                     message="stream read failed — check the URL/credentials, retrying"
                                     if state == "error" else "reconnecting")
                pipeline.set_state(Gst.State.NULL)
                self._gst_pipeline = None
                time.sleep(2)
                if stop_event.is_set():
                    break
                pipeline, sink = self._open_gst(cfg["rtsp_url"])
                continue

            frame = _sample_to_ndarray(sample)
            if frame is None:
                continue

            open_failures = 0
            last_frame_time = time.time()
            self._status.update(state="connected", rtsp_url=cfg["rtsp_url"], message=None,
                                 last_frame_at=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
            with self._latest_lock:
                self._latest_ref['frame'] = frame
                self._latest_ref['seq'] += 1
            self._last_frame_time = last_frame_time

        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        self._gst_pipeline = None

    def _watch_loop(self, stop_event):
        while not stop_event.is_set():
            time.sleep(3)
            if stop_event.is_set():
                break
            if self._last_frame_time and time.time() - self._last_frame_time > STALL_TIMEOUT:
                self._status["message"] = "no frames recently — forcing reconnect"
                # Unlike ffmpeg, GStreamer has no built-in socket timeout —
                # force the pipeline down so the capture loop's own
                # reconnect logic (above) picks it back up.
                pipeline = self._gst_pipeline
                if pipeline is not None:
                    pipeline.set_state(Gst.State.NULL)
                self._last_frame_time = time.time()  # avoid re-triggering while it reconnects

    def _infer_loop(self, stop_event, cfg):
        model = _get_model()
        last_seen_seq = -1
        class_conf = {0: cfg["fire_conf"], 1: cfg["smoke_conf"]}
        floor = min(cfg["fire_conf"], cfg["smoke_conf"])

        while not stop_event.is_set():
            with self._latest_lock:
                seq = self._latest_ref['seq']
                frame = self._latest_ref['frame']
            if frame is None or seq == last_seen_seq:
                time.sleep(0.005)
                continue
            last_seen_seq = seq
            frame = frame.copy()

            h, w = frame.shape[:2]
            if w > PROCESS_WIDTH:
                scale = PROCESS_WIDTH / w
                frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))

            results = model(frame, conf=floor, verbose=False)[0]
            for det in results.boxes:
                if stop_event.is_set():
                    break
                cls_id = int(det.cls[0])
                conf = float(det.conf[0])
                cls_name = CLASS_MAP.get(cls_id, f"class{cls_id}")
                required = class_conf.get(cls_id, floor)
                if conf < required:
                    continue
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                color = COLORS.get(cls_name, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                self._log_detection(cls_name, conf, frame, cfg)

            with self._annotated_lock:
                self._latest_annotated = frame

    def _log_detection(self, cls_name, conf, frame, cfg):
        now = time.time()
        if now - self._last_logged.get(cls_name, 0) < LOG_COOLDOWN_SECONDS:
            return
        self._last_logged[cls_name] = now

        ts = datetime.now(timezone.utc)
        footage_filename = f"{ts.strftime('%Y%m%d_%H%M%S')}_{cls_name}.jpg"
        cv2.imwrite(os.path.join(FOOTAGE_DIR, footage_filename), frame)

        from models import db, Detection
        with self._app.app_context():
            row = Detection(
                timestamp=ts.replace(tzinfo=None), location=cfg["location"],
                camera_name=cfg["camera_name"], detection_type=cls_name,
                confidence=conf, is_violation=True, footage_path=footage_filename,
            )
            db.session.add(row)
            db.session.commit()

            if cls_name == "fire":
                self._maybe_send_fire_alert(row.id, cfg)

    def _maybe_send_fire_alert(self, detection_id, cfg):
        """Called from inside the app-context block _log_detection already
        opened — same session, one commit."""
        from models import db, AlertRecipient, AlertEvent
        now = time.time()
        if now - self._last_alert_sent.get("fire", 0) < ALERT_SMS_COOLDOWN_SECONDS:
            return
        recipient = AlertRecipient.query.order_by(AlertRecipient.id.desc()).first()
        if not recipient:
            return
        message, status = send_fire_alert(recipient.phone_number, cfg["location"], cfg["camera_name"])
        db.session.add(AlertEvent(detection_id=detection_id, phone_number=recipient.phone_number,
                                   message=message, status=status))
        db.session.commit()
        self._last_alert_sent["fire"] = now


worker = CameraWorker()
