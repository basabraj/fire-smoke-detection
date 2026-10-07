import os
from dotenv import load_dotenv
load_dotenv()

import uuid
import cv2
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, send_file, Response, url_for, jsonify
from ultralytics import YOLO

from detection_logger import log_detection, FrameArchiver

# ── DISABLE GPU ───────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU globally

# ── CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'best.pt')
CAMERA_NAME = os.environ.get('CAMERA_NAME', 'Web-Upload')

# ── CLASS OVERRIDE ────────────────────────────────────────────────────────
# NOTE: this checkpoint's embedded model.names metadata says {0: 'Fire',
# 1: 'Smoke'} — but that metadata is WRONG for what the model actually
# learned. Tested directly against unambiguous reference photos (clear
# flame-only fire, clear industrial fire) and in both cases the network's
# class-1 predictions land squarely on the flames — i.e. the model was
# really trained with Smoke=0/Fire=1, and only the saved .pt names dict is
# mislabeled (a known way this can happen: names list order not matching
# the training data.yaml). Trust the empirical mapping below, not
# model.names.
CLASS_MAP = {
    0: 'Smoke',
    1: 'Fire'
}

COLORS = {
    'Fire': (0, 0, 139),   # BGR: deep red
    'Smoke': (139, 0, 0)   # BGR: deep blue
}

# YOLO's own default is 0.25 if you don't pass conf= — too permissive, lets
# weak/noisy boxes through (e.g. background clutter misread as "Smoke").
# Smoke's floor is set higher than Fire's — it's the one that's shown false
# positives (dark hair/skin texture misread as "Smoke") on live camera feeds.
FIRE_CONF = 0.6
SMOKE_CONF = 0.65
MODEL_CONF_FLOOR = min(FIRE_CONF, SMOKE_CONF)
CLASS_CONF = {0: SMOKE_CONF, 1: FIRE_CONF}  # keyed by the same id as CLASS_MAP

# ── APP SETUP ─────────────────────────────────────────────────────────────
app = Flask(__name__)
model = YOLO(MODEL_PATH)
model.to('cpu')
archiver = FrameArchiver(positive_interval=5)

# ── HOME ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ── IMAGE PROCESSING ──────────────────────────────────────────────────────
@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return "Image decode failed", 400

        clean_img = img.copy()  # pre-annotation, for archiving/retraining
        results = model(img, conf=MODEL_CONF_FLOOR)[0]
        detections = []  # (cls_id, conf, x1, y1, x2, y2) — only ones passing CLASS_CONF

        for det in results.boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            if conf < CLASS_CONF.get(cls_id, MODEL_CONF_FLOOR):
                continue
            detections.append((cls_id, conf, *map(int, det.xyxy[0])))
            cls_name = CLASS_MAP.get(cls_id, f"Class {cls_id}")
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            color = COLORS.get(cls_name, (255, 255, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{cls_name} {conf:.2f}"
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            label_x = x1
            label_y = y2 + text_size[1] + 5
            cv2.putText(img, text, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Only archive/log when something was actually detected — a frame
        # with nothing found is never written to disk.
        if detections:
            frame_path = archiver.maybe_save_positive(
                clean_img, [(c, x1, y1, x2, y2) for c, _, x1, y1, x2, y2 in detections],
                annotated_frame=img
            )
            for cls_id, conf, x1, y1, x2, y2 in detections:
                cls_name = CLASS_MAP.get(cls_id, f"Class {cls_id}")
                log_detection(CAMERA_NAME, cls_name, conf, (x1, y1, x2, y2), frame_path)

        _, buf = cv2.imencode('.png', img)
        return send_file(BytesIO(buf.tobytes()), mimetype='image/png')

    except Exception as e:
        return str(e), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    save_path = os.path.join(UPLOAD_DIR, 'live_input.mp4')
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file.save(save_path)

    app.config['CURRENT_VIDEO_PATH'] = save_path
    return jsonify({"stream_url": "/video_feed"})




# ── SERVE PROCESSED VIDEO FILE ────────────────────────────────────────────
@app.route('/video_file/<video_id>')
def serve_video(video_id):
    output_path = os.path.join(OUTPUT_DIR, f'{video_id}_output.mp4')
    return send_file(output_path, mimetype='video/mp4', as_attachment=False)

# ── VIDEO PROCESSING FUNCTION ─────────────────────────────────────────────
def gen_frames(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        results = model(frame, conf=MODEL_CONF_FLOOR)[0]
        for det in results.boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            if conf < CLASS_CONF.get(cls_id, MODEL_CONF_FLOOR):
                continue
            cls_name = CLASS_MAP.get(cls_id, f"Class {cls_id}")
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            color = COLORS.get(cls_name, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{cls_name} {conf:.2f}"
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            label_x = x1
            label_y = y2 + text_size[1] + 5
            cv2.putText(frame, text, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        out_writer.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    out_writer.release()

def gen_live_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        clean_frame = frame.copy()  # pre-annotation, for archiving/retraining
        results = model(frame, conf=MODEL_CONF_FLOOR)[0]
        detections = []  # (cls_id, conf, x1, y1, x2, y2) — only ones passing CLASS_CONF

        for det in results.boxes:
            cls_id = int(det.cls[0])
            conf = float(det.conf[0])
            if conf < CLASS_CONF.get(cls_id, MODEL_CONF_FLOOR):
                continue
            detections.append((cls_id, conf, *map(int, det.xyxy[0])))
            cls_name = CLASS_MAP.get(cls_id, f"Class {cls_id}")
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            color = COLORS.get(cls_name, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{cls_name} {conf:.2f}"
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            label_x = x1
            label_y = y2 + text_size[1] + 5
            cv2.putText(frame, text, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Only archive/log when something was actually detected — a frame
        # with nothing found is never written to disk.
        if detections:
            frame_path = archiver.maybe_save_positive(
                clean_frame, [(c, x1, y1, x2, y2) for c, _, x1, y1, x2, y2 in detections],
                annotated_frame=frame
            )
            for cls_id, conf, x1, y1, x2, y2 in detections:
                cls_name = CLASS_MAP.get(cls_id, f"Class {cls_id}")
                log_detection(CAMERA_NAME, cls_name, conf, (x1, y1, x2, y2), frame_path)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    video_path = app.config.get('CURRENT_VIDEO_PATH')
    return Response(gen_live_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_app():
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    run_app()
