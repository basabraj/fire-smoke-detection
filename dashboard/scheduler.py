"""
scheduler.py — the two automatic background jobs. Neither has a manual
trigger anywhere in the UI, per the spec: training only ever runs inside the
window set by config.TRAINING_WINDOW_START_HOUR/END_HOUR, and the "new
images?" scan is purely time-driven.
"""
import os
import glob
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

import config
from models import db, AnnotationImage, TrainingRun

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def scan_for_new_images(app):
    """Runs every SCAN_INTERVAL_HOURS. Walks the fire/ and smoke/ frames
    folders and registers any image not already tracked, so it shows up in
    the Training page's annotation queue on its own — this is the 'check
    every 1 hour if new images arrived' requirement. Also walks:
      - config.VALIDATION_HOLDOUT_FOLDERS, registered with is_holdout=True
        (shows up on the Training page's "Validation Set" tab instead of the
        regular Fire/Smoke queues) so they're pinned as fixed val data and
        never swept into the train pool.
      - config.NEGATIVE_WATCH_FOLDERS (no_fire/no_smoke), registered
        pre-annotated (no box to draw — an image with no label file is
        already a valid YOLO negative) so they feed straight into training
        without sitting in anyone's queue."""
    with app.app_context():
        for class_type, folder in config.WATCH_FOLDERS.items():
            if not os.path.isdir(folder):
                continue
            known = {r.image_path for r in AnnotationImage.query.filter_by(class_type=class_type).all()}
            added = 0
            for ext in IMAGE_EXTENSIONS:
                for path in glob.glob(os.path.join(folder, f"*{ext}")):
                    path = os.path.abspath(path)
                    if path in known:
                        continue
                    db.session.add(AnnotationImage(image_path=path, class_type=class_type))
                    added += 1
            if added:
                db.session.commit()
                print(f"[scanner] {class_type}: {added} new image(s) queued for annotation")

        for class_type, folder in config.VALIDATION_HOLDOUT_FOLDERS.items():
            if not os.path.isdir(folder):
                continue
            known = {r.image_path for r in AnnotationImage.query.filter_by(class_type=class_type, is_holdout=True).all()}
            added = 0
            for ext in IMAGE_EXTENSIONS:
                for path in glob.glob(os.path.join(folder, f"*{ext}")):
                    path = os.path.abspath(path)
                    if path in known:
                        continue
                    db.session.add(AnnotationImage(image_path=path, class_type=class_type, is_holdout=True))
                    added += 1
            if added:
                db.session.commit()
                print(f"[scanner] {class_type}: {added} new VALIDATION HOLDOUT image(s) queued for annotation")

        for class_type, folder in config.NEGATIVE_WATCH_FOLDERS.items():
            if not os.path.isdir(folder):
                continue
            known = {r.image_path for r in AnnotationImage.query.filter_by(class_type=class_type).all()}
            added = 0
            for ext in IMAGE_EXTENSIONS:
                for path in glob.glob(os.path.join(folder, f"*{ext}")):
                    path = os.path.abspath(path)
                    if path in known:
                        continue
                    db.session.add(AnnotationImage(
                        image_path=path, class_type=class_type,
                        annotated=True, annotated_at=datetime.utcnow(),
                    ))
                    added += 1
            if added:
                db.session.commit()
                print(f"[scanner] {class_type}: {added} new negative image(s) registered (training-ready)")


def _in_training_window(now=None):
    now = now or datetime.now()
    start, end = config.TRAINING_WINDOW_START_HOUR, config.TRAINING_WINDOW_END_HOUR
    return start <= now.hour < end


def _find_resumable_run():
    """Two ways a run can be left for resuming:
    'running'            — the previous call never reached its own
                            finally-block at all: killed outright (Jetson
                            crash, SSH drop, kill -9) instead of exiting
                            normally.
    'stopped_window_end' — exited cleanly because 08:00 was hit mid-run;
                            already meant to continue "next night", just not
                            wired up to actually do so before now.
    Either way, only usable if it got at least one checkpoint saved
    (train_job.py checkpoints every CHECKPOINT_EVERY epochs) — otherwise
    there's nothing on disk to resume from, so it's discarded instead."""
    stale_rows = (TrainingRun.query
                  .filter(TrainingRun.status.in_(["running", "stopped_window_end"]))
                  .order_by(TrainingRun.started_at.desc())
                  .all())
    if not stale_rows:
        return None

    latest, older = stale_rows[0], stale_rows[1:]
    resumable = None
    if latest.resume_checkpoint and os.path.exists(latest.resume_checkpoint) and latest.image_ids:
        resumable = latest
    else:
        latest.status = "failed"
        latest.finished_at = datetime.utcnow()
        latest.notes = ((latest.notes + " ") if latest.notes else "") + \
            "Interrupted before its first checkpoint — discarded, not resumable."

    # Older leftover rows predate this resume logic (or are from before it
    # ran once already) and never got a terminal status either — clean them
    # up the same way instead of leaving them stuck as "running" forever in
    # the training history.
    for row in older:
        row.status = "failed"
        row.finished_at = row.finished_at or datetime.utcnow()
        row.notes = ((row.notes + " ") if row.notes else "") + "Stale row from an old interrupted run."

    db.session.commit()
    return resumable


def run_nightly_training(app, respect_window=True):
    """Triggered at the start of the window by the cron job below. Stops
    itself if it somehow spills past the end of the window rather than
    running all day — but only when respect_window=True, which is the
    default for the automatic cron trigger. manual_train_now.py (the
    terminal-only operator escape hatch — there's still deliberately no
    button/endpoint for this in the UI) calls this with respect_window=False
    so a manual run trains all its epochs straight through regardless of
    the clock, instead of stopping after however many epochs happen to fit
    before the window's end.

    Before starting fresh, checks for an interrupted previous run with a
    saved checkpoint (see _find_resumable_run) and continues that instead —
    otherwise every crash/disconnect/window-cutoff would throw away however
    many epochs it had already done and restart the full count from
    BASE_CHECKPOINT."""
    with app.app_context():
        # Fetched fresh each call rather than pinned to a run like image_ids
        # — it's a small, deliberately hand-curated set that barely changes,
        # so there's no real reproducibility cost to always using whatever
        # is currently annotated in config.VALIDATION_HOLDOUT_FOLDERS. Empty
        # until images are added there and annotated on the Training page's
        # "Validation Set" tab — train_job.py falls back to the old by-index
        # 90/10 split when it is.
        holdout_val_rows = AnnotationImage.query.filter_by(is_holdout=True, annotated=True).all()

        resumable = _find_resumable_run()

        if resumable is not None:
            run = resumable
            ids = [int(i) for i in run.image_ids.split(",") if i]
            rows_by_id = {r.id: r for r in AnnotationImage.query.filter(AnnotationImage.id.in_(ids)).all()}
            annotation_rows = [rows_by_id[i] for i in ids if i in rows_by_id]
            start_epoch = run.epochs_completed
            resume_from = run.resume_checkpoint
            print(f"[training] resuming run #{run.id} from epoch {start_epoch}")
        else:
            # is_holdout images are never part of the train pool — they're
            # fixed validation data, pinned by hand rather than carved out
            # of this pool by index (see models.py's is_holdout docstring).
            annotated_unused = AnnotationImage.query.filter_by(
                annotated=True, used_in_training=False, is_holdout=False).all()

            if not annotated_unused:
                db.session.add(TrainingRun(
                    started_at=datetime.utcnow(), finished_at=datetime.utcnow(),
                    status="skipped_no_data", images_used=0,
                    notes="No newly annotated images since the last run.",
                ))
                db.session.commit()
                print("[training] skipped — no new annotated images")
                return

            annotation_rows = annotated_unused
            run = TrainingRun(
                started_at=datetime.utcnow(), status="running", images_used=len(annotation_rows),
                image_ids=",".join(str(r.id) for r in annotation_rows),
            )
            db.session.add(run)
            db.session.commit()
            start_epoch = 0
            resume_from = None

        try:
            # The actual fine-tune call. Left as a narrow, swappable
            # function so the scheduling/window logic above (the part this
            # spec cares about) doesn't depend on how training itself is
            # invoked.
            from train_job import fine_tune
            for _progress in fine_tune(annotation_rows, run, db, start_epoch=start_epoch,
                                        resume_from=resume_from, holdout_val_rows=holdout_val_rows):
                if respect_window and not _in_training_window():
                    run.status = "stopped_window_end"
                    run.notes = "Training window end reached; stopping and resuming next night."
                    break
            else:
                for row in annotation_rows:
                    row.used_in_training = True
                run.status = "completed"
        except Exception as e:
            run.status = "failed"
            run.notes = str(e)
        finally:
            run.finished_at = datetime.utcnow()
            db.session.commit()
            print(f"[training] run finished: {run.status}")


def start_scheduler(app):
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(scan_for_new_images, "interval", hours=config.SCAN_INTERVAL_HOURS,
                  args=[app], next_run_time=datetime.now(), id="hourly_scan")
    sched.add_job(run_nightly_training, "cron",
                  day_of_week=config.TRAINING_DAYS,
                  hour=config.TRAINING_WINDOW_START_HOUR, minute=0,
                  args=[app], id="nightly_training")
    sched.start()
    return sched
