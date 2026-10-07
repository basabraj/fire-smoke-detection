"""
models.py — SQLAlchemy models for the Fire & Smoke dashboard.

Detection      one row per confirmed fire/smoke detection event (fed by
               rtsp_stream.py's detection loop, or any other camera script
               that imports log_detection()).
AnnotationImage one row per raw frame discovered under
               dataset/fire_smoke/{fire,smoke}/frames — the hourly scanner
               (scheduler.py) creates these; the Training page lets a human
               draw the bounding box; annotating writes the YOLO label file
               and flips `annotated`.
TrainingRun    one row per nightly (00:00-08:00) automatic training attempt.
AlertRecipient the single Indian mobile number the emergency fire SMS goes
               to (saved from the Alert page). Saving a new one replaces it.
AlertEvent     one row per fire-alert SMS attempt (sent/simulated/failed) —
               the Alert page's history log.
CameraConfig   the single active camera's RTSP URL + per-class confidence
               thresholds (saved from the Settings page). Saving a new one
               replaces it and reconnects camera_worker immediately.
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Detection(db.Model):
    __tablename__ = "detections"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    location = db.Column(db.String(200), nullable=False)
    camera_name = db.Column(db.String(120), nullable=False)
    detection_type = db.Column(db.String(20), nullable=False)  # 'fire' | 'smoke'
    confidence = db.Column(db.Float, nullable=False)
    is_violation = db.Column(db.Boolean, default=True)
    # relative path under dashboard/static/footage/ — the saved annotated frame
    footage_path = db.Column(db.String(400), nullable=True)
    # set when a human flags this as a false positive — feeds it back into
    # training as a hard negative (see /api/detections/<id>/mark-wrong)
    reviewed_false_positive = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "location": self.location,
            "camera_name": self.camera_name,
            "detection_type": self.detection_type,
            "confidence": round(self.confidence, 3),
            "is_violation": self.is_violation,
            "footage_url": f"/static/footage/{self.footage_path}" if self.footage_path else None,
            "reviewed_false_positive": self.reviewed_false_positive,
        }


class AnnotationImage(db.Model):
    __tablename__ = "annotation_images"

    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(500), nullable=False, unique=True)  # absolute path on disk
    class_type = db.Column(db.String(20), nullable=False)  # 'fire' | 'smoke'
    discovered_at = db.Column(db.DateTime, default=datetime.utcnow)
    annotated = db.Column(db.Boolean, default=False, index=True)
    annotated_at = db.Column(db.DateTime, nullable=True)
    label_path = db.Column(db.String(500), nullable=True)  # YOLO .txt written next to the image
    used_in_training = db.Column(db.Boolean, default=False)
    # 'human' (drawn/edited on the Training page) | 'ai' (written directly by
    # a model-assisted auto-label pass, never opened for human review yet).
    # Only meaningful once annotated=True. /api/training/save always writes
    # 'human' — opening an AI box there to review/fix it is exactly what
    # promotes it to human-verified, so a saved edit should flip this.
    annotated_by = db.Column(db.String(10), nullable=False, default="human")
    # True for images placed in config.VALIDATION_HOLDOUT_FOLDERS — a fixed,
    # hand-picked set that must NEVER be split across train/val by frame
    # index (see fire-smoke-dataset-pipeline memory: a plain 90/10 index
    # split let the same source video land on both sides, so the model was
    # being "validated" on near-duplicate frames of what it just trained on
    # — inflating mAP50 without proving real generalization). Once any
    # is_holdout images are annotated, train_job.py uses them as the entire
    # val set and every other annotated image goes to train instead of
    # re-splitting by index.
    is_holdout = db.Column(db.Boolean, default=False, index=True)


class TrainingRun(db.Model):
    __tablename__ = "training_runs"

    id = db.Column(db.Integer, primary_key=True)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    finished_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(30), default="running")  # running|completed|failed|skipped_no_data|stopped_window_end
    images_used = db.Column(db.Integer, default=0)
    epochs_completed = db.Column(db.Integer, default=0)
    notes = db.Column(db.Text, nullable=True)

    # Overall metrics (ultralytics' box metrics from the post-train validation pass)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    map50 = db.Column(db.Float, nullable=True)
    map50_95 = db.Column(db.Float, nullable=True)
    inference_ms = db.Column(db.Float, nullable=True)  # per-image inference speed
    # Per-class breakdown, fire and smoke
    fire_precision = db.Column(db.Float, nullable=True)
    fire_recall = db.Column(db.Float, nullable=True)
    fire_map50 = db.Column(db.Float, nullable=True)
    smoke_precision = db.Column(db.Float, nullable=True)
    smoke_recall = db.Column(db.Float, nullable=True)
    smoke_map50 = db.Column(db.Float, nullable=True)
    # A few saved validation-prediction thumbnails (comma-separated filenames
    # under static/training_samples/) — the "annotated clips" strip.
    sample_images = db.Column(db.Text, nullable=True)

    # Resume support: a run checkpoints itself every 5 epochs (see
    # train_job.py) so an interruption (Jetson crash, SSH drop, or hitting
    # the nightly window's end) can pick back up instead of restarting the
    # full epoch count from the base checkpoint. image_ids pins the resumed
    # run to the *same* AnnotationImage rows (and therefore the same
    # train/val split) the interrupted attempt used. Both are cleared once
    # the run finishes all its epochs normally.
    resume_checkpoint = db.Column(db.String(500), nullable=True)
    image_ids = db.Column(db.Text, nullable=True)

    # Best-epoch tracking: each `model.train(epochs=1, ...)` call in the
    # fine_tune() loop is its own fresh Trainer, so Ultralytics' own
    # "best.pt" is only ever the best *within that single epoch* — trivially
    # itself — never the best across the whole run. best_map50 is this run's
    # running max mAP50 seen so far across all its epochs (persisted here so
    # a resumed run — see resume_checkpoint above — remembers it instead of
    # restarting the comparison from a fresh -1). Whichever epoch set this
    # value has its weights saved to train_job.TRUE_BEST_CHECKPOINT_PATH,
    # which is what actually gets deployed to BASE_CHECKPOINT at the end,
    # not just whatever the last epoch happened to produce.
    best_map50 = db.Column(db.Float, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "started_at": self.started_at.strftime("%Y-%m-%d %H:%M"),
            "finished_at": self.finished_at.strftime("%Y-%m-%d %H:%M") if self.finished_at else None,
            "status": self.status,
            "images_used": self.images_used,
            "epochs_completed": self.epochs_completed,
            "precision": self.precision, "recall": self.recall,
            "map50": self.map50, "map50_95": self.map50_95,
            "inference_ms": self.inference_ms,
            "fire": {"precision": self.fire_precision, "recall": self.fire_recall, "map50": self.fire_map50},
            "smoke": {"precision": self.smoke_precision, "recall": self.smoke_recall, "map50": self.smoke_map50},
            "sample_images": (self.sample_images or "").split(",") if self.sample_images else [],
        }


class AlertRecipient(db.Model):
    __tablename__ = "alert_recipients"

    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String(15), nullable=False)  # normalized "+91XXXXXXXXXX"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "phone_number": self.phone_number,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


class AlertEvent(db.Model):
    __tablename__ = "alert_events"

    id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey("detections.id"), nullable=True)
    phone_number = db.Column(db.String(15), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), nullable=False)  # 'sent' | 'simulated' | 'failed'
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "detection_id": self.detection_id,
            "phone_number": self.phone_number,
            "message": self.message,
            "status": self.status,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


class CameraConfig(db.Model):
    __tablename__ = "camera_config"

    id = db.Column(db.Integer, primary_key=True)
    rtsp_url = db.Column(db.String(500), nullable=False)
    location = db.Column(db.String(200), default="Home")
    camera_name = db.Column(db.String(120), default="RTSP-Cam-1")
    fire_conf = db.Column(db.Float, default=0.65)
    smoke_conf = db.Column(db.Float, default=0.60)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "rtsp_url": self.rtsp_url,
            "location": self.location,
            "camera_name": self.camera_name,
            "fire_conf": self.fire_conf,
            "smoke_conf": self.smoke_conf,
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M:%S") if self.updated_at else None,
        }
