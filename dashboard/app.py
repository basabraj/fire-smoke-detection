"""
app.py — Fire & Smoke Detection Dashboard (Flask).

Run:  python app.py
Then open http://localhost:5050/
"""
import os
import re
import time
import shutil
import calendar
from datetime import datetime, timedelta, date

from io import BytesIO

import cv2
from flask import Flask, render_template, jsonify, request, send_from_directory, send_file, Response

import config
from models import db, Detection, AnnotationImage, TrainingRun, AlertRecipient, AlertEvent, CameraConfig
from scheduler import start_scheduler
from camera_worker import worker as camera_worker

INDIAN_MOBILE_RE = re.compile(r"^(?:\+?91)?([6-9]\d{9})$")


def normalize_indian_phone(raw):
    """'+919876543210' / '919876543210' / '9876543210' (spaces/dashes ok) ->
    '+919876543210', or None if it isn't a valid 10-digit Indian mobile
    number (must start 6-9, per TRAI numbering)."""
    cleaned = (raw or "").strip().replace(" ", "").replace("-", "")
    m = INDIAN_MOBILE_RE.match(cleaned)
    return f"+91{m.group(1)}" if m else None

app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)


def _ensure_training_run_columns():
    """create_all() only creates missing TABLES, not columns added to an
    already-existing one — this project has no Alembic migration set up, so
    add any new TrainingRun columns by hand the first time this runs against
    an older dashboard.db (see models.py: resume_checkpoint/image_ids,
    added for mid-run resume support; best_map50, added for true-best-epoch
    tracking)."""
    from sqlalchemy import text
    with db.engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(training_runs)"))}
        for col, ddl in (
            ("resume_checkpoint", "ALTER TABLE training_runs ADD COLUMN resume_checkpoint VARCHAR(500)"),
            ("image_ids", "ALTER TABLE training_runs ADD COLUMN image_ids TEXT"),
            ("best_map50", "ALTER TABLE training_runs ADD COLUMN best_map50 FLOAT"),
        ):
            if col not in existing:
                conn.execute(text(ddl))
        conn.commit()


def _ensure_annotation_image_columns():
    """Same create_all() limitation as above, for AnnotationImage's
    annotated_by column (see models.py). SQLite backfills every existing row
    with the column default ('human') when it's first added — correct for
    everything actually drawn on the Training page, but the 2026-09-29
    model-assisted fire auto-label batch needs re-marking 'ai' right after,
    identified by the single shared annotated_at timestamp that batch commit
    wrote (see the "reduce my work" auto-label pass)."""
    from sqlalchemy import text
    with db.engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(annotation_images)"))}
        if "annotated_by" not in existing:
            conn.execute(text("ALTER TABLE annotation_images ADD COLUMN annotated_by VARCHAR(10) NOT NULL DEFAULT 'human'"))
            conn.execute(text(
                "UPDATE annotation_images SET annotated_by='ai' "
                "WHERE annotated_at='2026-09-29 07:10:23.557834'"
            ))
        if "is_holdout" not in existing:
            conn.execute(text("ALTER TABLE annotation_images ADD COLUMN is_holdout BOOLEAN NOT NULL DEFAULT 0"))
        conn.commit()


with app.app_context():
    db.create_all()
    _ensure_training_run_columns()
    _ensure_annotation_image_columns()


# ── Page routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("overview.html", active="overview")


@app.route("/alert")
def alert_page():
    return render_template("alert.html", active="alert")


@app.route("/training")
def training_page():
    return render_template("training.html", active="training")


@app.route("/settings")
def settings_page():
    return render_template("settings.html", active="settings")


# ── Overview: stat cards ────────────────────────────────────────────────────
def _count(detection_type, since):
    q = Detection.query.filter(Detection.detection_type == detection_type)
    if since is not None:
        q = q.filter(Detection.timestamp >= since)
    return q.count()


@app.route("/api/stats")
def api_stats():
    now = datetime.utcnow()
    today_start = datetime(now.year, now.month, now.day)
    week_start = today_start - timedelta(days=today_start.weekday())  # Monday
    month_start = datetime(now.year, now.month, 1)

    return jsonify({
        "fire_today": _count("fire", today_start),
        "smoke_today": _count("smoke", today_start),
        "fire_week": _count("fire", week_start),
        "smoke_week": _count("smoke", week_start),
        "fire_month": _count("fire", month_start),
        "smoke_month": _count("smoke", month_start),
    })


# ── Overview: recent detections table ───────────────────────────────────────
@app.route("/api/table")
def api_table():
    limit = int(request.args.get("limit", 50))
    rows = (Detection.query
            .order_by(Detection.timestamp.desc())
            .limit(limit)
            .all())
    return jsonify([r.to_dict() for r in rows])


# ── Flag a detection for manual review (was: auto hard-negative) ───────────
@app.route("/api/detections/<int:detection_id>/mark-wrong", methods=["POST"])
def api_mark_detection_wrong(detection_id):
    """Flags a detection as needing a human look: copies its footage image
    into config.FLAGGED_REVIEW_DIR and registers an un-annotated
    AnnotationImage row (class_type='review') — it shows up on the Training
    page's 'Flagged for review' tab, where a person picks Fire or Smoke and
    draws the correct box (or leaves it empty to confirm it's really
    nothing). No label is guessed automatically here: a wrong box guessed
    by the model is exactly what got it flagged in the first place, so
    auto-writing a label from that same model output would just teach it
    the same mistake again."""
    d = Detection.query.get_or_404(detection_id)
    if d.reviewed_false_positive:
        return jsonify({"ok": True, "already": True})
    if not d.footage_path:
        return jsonify({"ok": False, "error": "No footage image saved for this detection."}), 400

    src = os.path.join(app.root_path, "static", "footage", d.footage_path)
    if not os.path.exists(src):
        return jsonify({"ok": False, "error": "Footage image file is missing on disk."}), 400

    review_folder = config.FLAGGED_REVIEW_DIR
    os.makedirs(review_folder, exist_ok=True)

    base = f"review_{d.detection_type}_{d.id}_{os.path.splitext(d.footage_path)[0]}"
    img_dst = os.path.abspath(os.path.join(review_folder, base + ".jpg"))
    shutil.copy2(src, img_dst)

    db.session.add(AnnotationImage(image_path=img_dst, class_type="review"))
    d.reviewed_false_positive = True
    db.session.commit()
    return jsonify({"ok": True})


# ── Overview: weekly-within-month bar/line chart (fire or smoke) ───────────
def _week_of_month_bounds(year, month):
    """Split a calendar month into 7-day 'weeks' (1-7, 8-14, ...) — the same
    week-row logic a calendar grid uses, which is what makes the prev/next
    month toggle behave like flipping a calendar page instead of an
    ISO-week count that wouldn't align to a single month."""
    days_in_month = calendar.monthrange(year, month)[1]
    bounds = []
    day = 1
    while day <= days_in_month:
        end_day = min(day + 6, days_in_month)
        start = datetime(year, month, day)
        end = datetime(year, month, end_day) + timedelta(days=1)  # exclusive
        bounds.append((f"Week {len(bounds)+1}", start, end))
        day = end_day + 1
    return bounds


@app.route("/api/chart/weekly")
def api_chart_weekly():
    detection_type = request.args.get("type", "fire")
    month_str = request.args.get("month")  # "YYYY-MM"
    if month_str:
        year, month = map(int, month_str.split("-"))
    else:
        today = date.today()
        year, month = today.year, today.month

    labels, counts = [], []
    for label, start, end in _week_of_month_bounds(year, month):
        c = (Detection.query
             .filter(Detection.detection_type == detection_type)
             .filter(Detection.timestamp >= start, Detection.timestamp < end)
             .count())
        labels.append(label)
        counts.append(c)

    prev_month = (datetime(year, month, 1) - timedelta(days=1))
    next_month = (datetime(year, month, 28) + timedelta(days=7)).replace(day=1)

    return jsonify({
        "labels": labels,
        "counts": counts,
        "month_label": datetime(year, month, 1).strftime("%B %Y"),
        "current": f"{year:04d}-{month:02d}",
        "prev": f"{prev_month.year:04d}-{prev_month.month:02d}",
        "next": f"{next_month.year:04d}-{next_month.month:02d}",
        "is_current_month": (year == date.today().year and month == date.today().month),
    })


# ── Overview: monthly-within-year bar chart (fire or smoke) ────────────────
@app.route("/api/chart/monthly")
def api_chart_monthly():
    detection_type = request.args.get("type", "fire")
    year = int(request.args.get("year", date.today().year))

    labels, counts = [], []
    for m in range(1, 13):
        start = datetime(year, m, 1)
        end = datetime(year + 1, 1, 1) if m == 12 else datetime(year, m + 1, 1)
        c = (Detection.query
             .filter(Detection.detection_type == detection_type)
             .filter(Detection.timestamp >= start, Detection.timestamp < end)
             .count())
        labels.append(calendar.month_abbr[m])
        counts.append(c)

    return jsonify({
        "labels": labels,
        "counts": counts,
        "year_label": str(year),
        "current": str(year),
        "prev": str(year - 1),
        "next": str(year + 1),
        "is_current_year": (year == date.today().year),
    })


# ── Overview: fire vs smoke split (pie chart) ───────────────────────────────
@app.route("/api/chart/split")
def api_chart_split():
    """Today's fire-vs-smoke proportion — satisfies the 'at least one pie
    chart' requirement. A pie is the wrong shape for a trend (see
    choosing-a-form.md), so it's reserved for this one proportion-of-a-whole
    question rather than reused for the week/month/year series above."""
    now = datetime.utcnow()
    today_start = datetime(now.year, now.month, now.day)
    fire = _count("fire", today_start)
    smoke = _count("smoke", today_start)
    return jsonify({"labels": ["Fire", "Smoke"], "counts": [fire, smoke]})


# ── Training: annotation queue + save ───────────────────────────────────────
@app.route("/api/training/queue")
def api_training_queue():
    """type=validation is the fixed holdout set (config.VALIDATION_HOLDOUT_FOLDERS,
    is_holdout=True) — combined fire+smoke into one queue like the "Flagged
    for review" tab, since it's a single small annotation job rather than two
    separate ones. Every other type is a regular queue and explicitly
    excludes holdout images (is_holdout=False) so they never show up mixed
    into the normal Fire/Smoke annotation work — they're fixed validation
    data, not part of the pool a person annotates for training."""
    class_type = request.args.get("type", "fire")
    if class_type == "validation":
        rows = (AnnotationImage.query
                .filter_by(is_holdout=True, annotated=False)
                .order_by(AnnotationImage.discovered_at.asc())
                .limit(200)
                .all())
    else:
        rows = (AnnotationImage.query
                .filter_by(class_type=class_type, annotated=False, is_holdout=False)
                .order_by(AnnotationImage.discovered_at.asc())
                .limit(200)
                .all())
    return jsonify([{
        "id": r.id,
        "filename": os.path.basename(r.image_path),
        "discovered_at": r.discovered_at.strftime("%Y-%m-%d %H:%M"),
        "class_type": r.class_type,
    } for r in rows])


@app.route("/api/training/counts")
def api_training_counts():
    def counts_for(class_type):
        # is_holdout=False so a validation-holdout image (which carries a
        # real 'fire'/'smoke' class_type, unlike 'review') never counts
        # towards the regular Fire/Smoke queue's pending/annotated totals.
        total = AnnotationImage.query.filter_by(class_type=class_type, is_holdout=False).count()
        pending = AnnotationImage.query.filter_by(class_type=class_type, is_holdout=False, annotated=False).count()
        return {"total": total, "pending": pending, "annotated": total - pending}

    validation_total = AnnotationImage.query.filter_by(is_holdout=True).count()
    validation_pending = AnnotationImage.query.filter_by(is_holdout=True, annotated=False).count()

    last_run = TrainingRun.query.order_by(TrainingRun.started_at.desc()).first()
    return jsonify({
        "fire": counts_for("fire"),
        "smoke": counts_for("smoke"),
        "review": counts_for("review"),
        "validation": {
            "total": validation_total, "pending": validation_pending,
            "annotated": validation_total - validation_pending,
        },
        "last_run": {
            "started_at": last_run.started_at.strftime("%Y-%m-%d %H:%M") if last_run else None,
            "status": last_run.status if last_run else "never",
            "images_used": last_run.images_used if last_run else 0,
        } if last_run else None,
        "window": f"{config.TRAINING_WINDOW_START_HOUR:02d}:00 - {config.TRAINING_WINDOW_END_HOUR:02d}:00",
    })


@app.route("/api/training/image/<int:image_id>")
def api_training_image(image_id):
    row = AnnotationImage.query.get_or_404(image_id)
    directory = os.path.dirname(row.image_path)
    filename = os.path.basename(row.image_path)

    # ?boxes=1 (Trained Images page thumbnails/preview) draws the saved
    # label's box(es) on top, since a plain thumbnail gives no visual way to
    # tell an annotated image apart from an unannotated one.
    if request.args.get("boxes") == "1" and row.label_path and os.path.exists(row.label_path):
        img = cv2.imread(row.image_path)
        h, w = img.shape[:2]
        color = (0, 0, 220) if row.class_type == "fire" else (220, 0, 0)  # BGR: red=fire, blue=smoke
        with open(row.label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                _cls, xc, yc, bw, bh = map(float, parts)
                x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, w // 200))
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")

    return send_from_directory(directory, filename)


@app.route("/api/training/results")
def api_training_results():
    """Run history table — the Ultralytics-benchmark-style 'All metrics'
    table on the Training page. Most recent first."""
    runs = TrainingRun.query.order_by(TrainingRun.started_at.desc()).limit(20).all()
    return jsonify([r.to_dict() for r in runs])


@app.route("/api/training/latest")
def api_training_latest():
    """Most recent COMPLETED run — feeds the per-class Precision/Recall and
    mAP bar charts, plus the annotated sample-prediction thumbnail strip."""
    run = (TrainingRun.query
           .filter_by(status="completed")
           .order_by(TrainingRun.started_at.desc())
           .first())
    return jsonify(run.to_dict() if run else None)


@app.route("/api/training/save", methods=["POST"])
def api_training_save():
    """Save a manually-drawn bounding box as a YOLO-format label file next
    to the source image. Class id: 0 = fire, 1 = smoke (matches the
    yolo_smoke_fire.pt checkpoint's CLASS_MAP already used elsewhere in this
    project).

    `class_type` in the request body is only sent by the "Flagged for
    review" tab, where the class isn't fixed by the queue — the person
    picks Fire or Smoke per image. The regular Fire/Smoke queue tabs don't
    send it, so this falls back to the row's already-locked class_type."""
    data = request.get_json(force=True)
    image_id = data["image_id"]
    boxes = data["boxes"]  # list of {x_center, y_center, width, height} normalized 0-1
    chosen_class = data.get("class_type")

    row = AnnotationImage.query.get_or_404(image_id)
    if chosen_class in ("fire", "smoke"):
        row.class_type = chosen_class
    class_id = 0 if row.class_type == "fire" else 1

    label_path = os.path.splitext(row.image_path)[0] + ".txt"
    with open(label_path, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(f"{class_id} {b['x_center']:.6f} {b['y_center']:.6f} "
                     f"{b['width']:.6f} {b['height']:.6f}\n")

    row.annotated = True
    row.annotated_at = datetime.utcnow()
    row.label_path = label_path
    # Always 'human' here — this endpoint only ever runs from the Training
    # page's own UI (a fresh box, or reviewing/fixing an 'ai' one via the
    # Trained Images page's Edit button), and either way that's a human
    # verifying the label now.
    row.annotated_by = "human"
    db.session.commit()
    return jsonify({"ok": True})


@app.route("/api/training/boxes/<int:image_id>")
def api_training_boxes(image_id):
    """Reads an already-saved YOLO label file back out as normalized boxes,
    for the Trained Images page's Edit view — /api/training/save only ever
    writes labels, nothing previously read them back for re-editing."""
    row = AnnotationImage.query.get_or_404(image_id)
    boxes = []
    if row.label_path and os.path.exists(row.label_path):
        with open(row.label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    _cls, xc, yc, bw, bh = parts
                    boxes.append({"x_center": float(xc), "y_center": float(yc),
                                  "width": float(bw), "height": float(bh)})
    return jsonify({
        "id": row.id,
        "filename": os.path.basename(row.image_path),
        "class_type": row.class_type,
        "boxes": boxes,
    })


# ── Trained Images: already-trained / annotated-by-user / not-yet-annotated ─
@app.route("/trained-images")
def trained_images_page():
    return render_template("trained_images.html", active="trained_images")


@app.route("/api/trained/counts")
def api_trained_counts():
    def counts_for(class_type):
        return {
            "already_trained": AnnotationImage.query.filter_by(
                class_type=class_type, annotated=True, used_in_training=True).count(),
            "annotated_by_user": AnnotationImage.query.filter_by(
                class_type=class_type, annotated=True, used_in_training=False).count(),
            "not_annotated": AnnotationImage.query.filter_by(
                class_type=class_type, annotated=False).count(),
        }
    return jsonify({"fire": counts_for("fire"), "smoke": counts_for("smoke")})


@app.route("/api/trained/list")
def api_trained_list():
    """Backs all three Trained Images sections — `section` picks which slice
    of AnnotationImage this is:
      already_trained    — annotated=True, used_in_training=True
      annotated_by_user  — annotated=True, used_in_training=False (done, but
                            not yet swept into a completed training run)
      not_annotated      — annotated=False (same rows as the Training page's
                            own queue, just browsable here instead of one at
                            a time)
    """
    section = request.args.get("section", "annotated_by_user")
    class_type = request.args.get("type", "fire")
    limit = min(int(request.args.get("limit", 150)), 150)
    offset = max(int(request.args.get("offset", 0)), 0)

    q = AnnotationImage.query.filter_by(class_type=class_type)
    if section == "already_trained":
        q = q.filter_by(annotated=True, used_in_training=True).order_by(AnnotationImage.annotated_at.desc())
    elif section == "not_annotated":
        q = q.filter_by(annotated=False).order_by(AnnotationImage.discovered_at.asc())
    else:  # annotated_by_user
        q = q.filter_by(annotated=True, used_in_training=False).order_by(AnnotationImage.annotated_at.desc())

    total = q.count()
    rows = q.offset(offset).limit(limit).all()
    return jsonify({
        "total": total,
        "offset": offset,
        "rows": [{
            "id": r.id,
            "filename": os.path.basename(r.image_path),
            "when": (r.annotated_at or r.discovered_at).strftime("%Y-%m-%d %H:%M"),
            "annotated_by": r.annotated_by if r.annotated else None,
            "has_box": bool(r.label_path and os.path.exists(r.label_path)),
        } for r in rows],
    })


# ── Alert: emergency contact + fire gallery + SMS history ──────────────────
@app.route("/api/alert/recipient", methods=["GET"])
def api_alert_recipient_get():
    row = AlertRecipient.query.order_by(AlertRecipient.id.desc()).first()
    return jsonify(row.to_dict() if row else None)


@app.route("/api/alert/recipient", methods=["POST"])
def api_alert_recipient_save():
    data = request.get_json(force=True)
    phone = normalize_indian_phone(data.get("phone_number"))
    if not phone:
        return jsonify({"ok": False, "error": "Enter a valid 10-digit Indian mobile number."}), 400
    AlertRecipient.query.delete()  # single active recipient — a new save replaces it
    db.session.add(AlertRecipient(phone_number=phone))
    db.session.commit()
    return jsonify({"ok": True, "phone_number": phone})


@app.route("/api/alert/recipient", methods=["DELETE"])
def api_alert_recipient_delete():
    AlertRecipient.query.delete()
    db.session.commit()
    return jsonify({"ok": True})


@app.route("/api/alert/fire-images")
def api_alert_fire_images():
    limit = int(request.args.get("limit", 24))
    rows = (Detection.query
            .filter(Detection.detection_type == "fire")
            .order_by(Detection.timestamp.desc())
            .limit(limit)
            .all())
    return jsonify([r.to_dict() for r in rows])


@app.route("/api/alert/history")
def api_alert_history():
    limit = int(request.args.get("limit", 20))
    rows = AlertEvent.query.order_by(AlertEvent.created_at.desc()).limit(limit).all()
    return jsonify([r.to_dict() for r in rows])


# ── Settings: camera RTSP + confidence, wired straight to camera_worker ────
@app.route("/api/camera/config", methods=["GET"])
def api_camera_config_get():
    row = CameraConfig.query.order_by(CameraConfig.id.desc()).first()
    return jsonify(row.to_dict() if row else None)


@app.route("/api/camera/config", methods=["POST"])
def api_camera_config_save():
    data = request.get_json(force=True)
    rtsp_url = (data.get("rtsp_url") or "").strip()
    if not rtsp_url:
        return jsonify({"ok": False, "error": "RTSP URL is required."}), 400
    try:
        fire_conf = float(data.get("fire_conf", 0.65))
        smoke_conf = float(data.get("smoke_conf", 0.60))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Confidence values must be numbers."}), 400
    if not (0 < fire_conf <= 1 and 0 < smoke_conf <= 1):
        return jsonify({"ok": False, "error": "Confidence must be between 0 and 1."}), 400
    location = (data.get("location") or "Home").strip()
    camera_name = (data.get("camera_name") or "RTSP-Cam-1").strip()

    CameraConfig.query.delete()  # single active camera — a new save replaces it
    row = CameraConfig(rtsp_url=rtsp_url, location=location, camera_name=camera_name,
                        fire_conf=fire_conf, smoke_conf=smoke_conf)
    db.session.add(row)
    db.session.commit()

    camera_worker.start(app, rtsp_url, location, camera_name, fire_conf, smoke_conf)
    return jsonify({"ok": True, **row.to_dict()})


@app.route("/api/camera/status")
def api_camera_status():
    return jsonify(camera_worker.get_status())


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            jpeg = camera_worker.get_jpeg()
            if jpeg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            else:
                time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    start_scheduler(app)
    with app.app_context():
        cam = CameraConfig.query.order_by(CameraConfig.id.desc()).first()
        if cam:
            camera_worker.start(app, cam.rtsp_url, cam.location, cam.camera_name, cam.fire_conf, cam.smoke_conf)
    try:
        app.run(host="0.0.0.0", port=5050, debug=False)
    finally:
        # Join the camera worker's CUDA-using threads before the interpreter
        # tears down — see the docstring on CameraWorker.stop() for why.
        camera_worker.stop()
