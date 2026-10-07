import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset", "fire_smoke"))

# Where the hourly scanner looks for new raw frames to queue for annotation.
WATCH_FOLDERS = {
    "fire": os.path.join(DATASET_ROOT, "fire", "frames"),
    "smoke": os.path.join(DATASET_ROOT, "smoke", "frames"),
}

# Negative (background-only, no fire/no smoke) frames — these need no human
# annotation (an image with no label file is a valid YOLO negative example),
# so the scanner registers them straight in as annotated=True instead of
# queuing them on the Training page.
NEGATIVE_WATCH_FOLDERS = {
    "no_fire": os.path.join(DATASET_ROOT, "no_fire", "frames"),
    "no_smoke": os.path.join(DATASET_ROOT, "no_smoke", "frames"),
}

# Flagged live detections land here (see /api/detections/<id>/mark-wrong) for
# manual review on the Training page's "Flagged for review" tab — kept out of
# WATCH_FOLDERS so the hourly scanner never re-discovers them on its own; the
# flag endpoint registers the AnnotationImage row itself.
FLAGGED_REVIEW_DIR = os.path.join(DATASET_ROOT, "flagged_review")

# A fixed, hand-picked validation set — drop images here (never sourced from
# the same videos as WATCH_FOLDERS) and annotate them on the Training page's
# "Validation Set" tab. The hourly scanner registers them with is_holdout=True
# (see models.py), which permanently excludes them from the train pool;
# train_job.py uses them as the entire val split instead of carving 10% out
# of the regular pool by list index — the old approach let the same source
# video land on both sides of that split, inflating validation scores with
# near-duplicate frames of what the model had just trained on.
VALIDATION_HOLDOUT_FOLDERS = {
    "fire": os.path.join(DATASET_ROOT, "validation_holdout", "fire", "frames"),
    "smoke": os.path.join(DATASET_ROOT, "validation_holdout", "smoke", "frames"),
}

SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "dashboard.db")
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Auto-training window — no manual trigger exists anywhere in the UI.
TRAINING_WINDOW_START_HOUR = 14  # 14:00 (2:00 PM)
TRAINING_WINDOW_END_HOUR = 18    # 18:00 (6:00 PM)
# APScheduler cron day_of_week syntax: Sunday off, so a full week's camera
# footage still has one day with nothing competing with camera_worker for
# the Jetson's RAM/GPU.
TRAINING_DAYS = "mon-sat"

# How often the "new images?" scan runs.
SCAN_INTERVAL_HOURS = 1

# Emergency SMS alert — minimum gap between two fire-alert texts to the same
# recipient, so one ongoing fire (many detections in a row) doesn't flood
# their phone. Actual sending is a stub until an SMS provider API key is
# configured — see dashboard/sms_alert.py.
ALERT_SMS_COOLDOWN_SECONDS = 300  # 5 minutes

SECRET_KEY = "dev-key-change-in-production"
