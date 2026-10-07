import time, threading,gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np

Gst.init(None)

RTSP_URL = "rtsp://admin:Digital@123@192.168.96.83:554/live1.sdp"

# If no frame arrives for this long, assume the pipeline is stuck and force a reconnect
STALL_TIMEOUT = 10  # seconds


PIPELINE_DESC = (
    f'rtspsrc location="{RTSP_URL}" latency=200 protocols=tcp tcp-timeout=10000000 ! '
    'rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! '
    'video/x-raw,format=BGR ! '
    'appsink name=sink max-buffers=1 drop=true sync=false'
)

last_frame_time = time.time()
pipeline_ref = {'pipeline': None}
stop_event = threading.Event()


def build_pipeline():
    """Build and start the pipeline. Fails loudly (fix #2) instead of
    returning a half-built pipeline with None elements in it."""
    try:
        pipeline = Gst.parse_launch(PIPELINE_DESC)
    except GLib.Error as e:
        raise SystemExit(
            f"[FATAL] Could not build pipeline — check that all required "
            f"GStreamer plugins are installed (gstreamer1.0-plugins-bad, "
            f"gstreamer1.0-libav): {e}"
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


# ── Watchdog Thread (fix #4) ─────────────────────────────────────────────

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


# ── Bus watcher (fix #3 — surface link/negotiation errors instead of
# swallowing them) ────────────────────────────────────────────────────────
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


# ── Detection/pull loop with reconnect (fix #4) ───────────────────────────
def detect_loop():
    global last_frame_time
    pipeline, sink = build_pipeline()
    frame_count = 0

    while not stop_event.is_set():
        sample = sink.emit('pull-sample')
        if sample is None:
            # EOS, error, or force-killed by the watchdog — reconnect
            print("[WARN] Pipeline produced no sample, reconnecting...")
            pipeline.set_state(Gst.State.NULL)
            time.sleep(2)
            pipeline, sink = build_pipeline()
            continue

        frame = sample_to_ndarray(sample)
        if frame is None:
            continue

        last_frame_time = time.time()
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[OK] {frame_count} frames received so far, latest shape={frame.shape}")


if __name__ == '__main__':
    threading.Thread(target=detect_loop, daemon=True).start()
    threading.Thread(target=watch_for_stall, daemon=True).start()
    threading.Thread(target=bus_watcher, daemon=True).start()

    print("Stream test started (headless-safe, auto-reconnecting). Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        pipeline = pipeline_ref.get('pipeline')
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
