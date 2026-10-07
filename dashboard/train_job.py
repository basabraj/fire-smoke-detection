"""
train_job.py — the actual YOLO fine-tune call, kept separate from
scheduler.py so the nightly-window logic there stays simple. Builds a
temporary YOLO dataset from whatever's been annotated since the last run,
fine-tunes the existing yolo_smoke_fire.pt checkpoint (transfer learning,
never from scratch — see fire-smoke-dataset-pipeline memory), then runs a
validation pass and saves the metrics + a few annotated sample predictions
onto the TrainingRun row so the Training page can show an Ultralytics-
benchmark-style result (per-class bar charts + a metrics table + a sample
thumbnail strip).
"""
import os
import shutil
import tempfile

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HOME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CHECKPOINT = os.path.join(HOME_DIR, "yolo_smoke_fire.pt")
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "training_samples")

# Resume support: every CHECKPOINT_EVERY epochs, the in-progress weights get
# copied here and the path recorded on the TrainingRun row (see
# scheduler.py). If the process dies (Jetson crash, SSH drop, hitting the
# nightly window's end) before finishing all its epochs, the next call picks
# this file back up instead of restarting from BASE_CHECKPOINT — losing at
# most CHECKPOINT_EVERY-1 epochs of progress instead of the whole run.
CHECKPOINT_EVERY = 5
RESUME_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "training_runs", "nightly_resume.pt"
)

# True-best-epoch tracking: each model.train(epochs=1, ...) call below is its
# own fresh Trainer, so Ultralytics' own weights/best.pt is only ever "best"
# within that single epoch — trivially itself, never the best across the
# whole run (confirmed on a real run: epoch 1 hit mAP50=0.603, the final
# epoch 20 only 0.472, and epoch 20's weights were what got deployed simply
# because it ran last). TRUE_BEST_CHECKPOINT_PATH holds whichever epoch's
# weights actually had the best mAP50 seen so far, and that's what gets
# deployed to BASE_CHECKPOINT at the end instead.
TRUE_BEST_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "training_runs", "true_best.pt"
)


def _copy_row(row, dataset_dir, split):
    # Source frames get extracted into per-class folders (fire/, no_fire/,
    # smoke/, no_smoke/) independently, so the same basename can turn up
    # in more than one of them (confirmed: 54 collisions between fire/
    # and no_fire/ alone). Prefixing with the row id keeps every
    # destination filename unique so one class's frame never silently
    # overwrites another's image or label in the merged dataset.
    stem = f"{row.id}_{os.path.splitext(os.path.basename(row.image_path))[0]}"
    ext = os.path.splitext(row.image_path)[1]
    img_dst = os.path.join(dataset_dir, split, "images", stem + ext)
    lbl_dst = os.path.join(dataset_dir, split, "labels", stem + ".txt")
    shutil.copy2(row.image_path, img_dst)
    if row.label_path and os.path.exists(row.label_path):
        shutil.copy2(row.label_path, lbl_dst)


def _build_yolo_dataset(annotation_rows, dataset_dir, holdout_rows=None):
    """holdout_rows (config.VALIDATION_HOLDOUT_FOLDERS, annotated on the
    Training page's "Validation Set" tab) — when there are any, they become
    the *entire* val split and every row in annotation_rows goes to train
    (100/0, no carving val out of the train pool at all). Without them,
    falls back to the old by-list-index 90/10 split.

    That fallback is what a plain index split on annotation_rows always was
    here, and it has a real bug for video-frame data: consecutive frames
    from the same source video land a few seconds apart in the list, so an
    index cut routinely put some frames from a video in train and others
    from the *same* video in val (confirmed on a real run: 22 of 24 val-set
    source videos also had frames in train) — validation was measuring
    near-duplicate recognition, not generalization, and inflating mAP50.
    A held-out set from videos that never contribute to train doesn't have
    this problem regardless of how it's split internally (it isn't)."""
    for split in ("train", "val"):
        os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split, "labels"), exist_ok=True)

    if holdout_rows:
        for row in annotation_rows:
            _copy_row(row, dataset_dir, "train")
        for row in holdout_rows:
            _copy_row(row, dataset_dir, "val")
    else:
        n_val = max(1, len(annotation_rows) // 10)
        for i, row in enumerate(annotation_rows):
            _copy_row(row, dataset_dir, "val" if i < n_val else "train")

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            "train: train/images\n"
            "val: val/images\n"
            "nc: 2\n"
            "names: ['fire', 'smoke']\n"
        )
    return yaml_path, os.path.join(dataset_dir, "val", "images")


def _extract_metrics(val_results, model):
    """Pull the numbers the Training page's charts/table need out of
    ultralytics' validation result object. Returns a plain dict — never the
    ultralytics object itself, so callers don't need that import."""
    box = val_results.box
    names = model.names  # {0: 'fire', 1: 'smoke'}
    class_ids = {v: k for k, v in names.items()}

    def per_class(cls_name, attr, fallback=None):
        idx = class_ids.get(cls_name)
        try:
            arr = getattr(box, attr)
            return float(arr[idx]) if idx is not None and idx < len(arr) else fallback
        except Exception:
            return fallback

    return {
        "precision": float(box.mp) if hasattr(box, "mp") else None,
        "recall": float(box.mr) if hasattr(box, "mr") else None,
        "map50": float(box.map50) if hasattr(box, "map50") else None,
        "map50_95": float(box.map) if hasattr(box, "map") else None,
        "inference_ms": float(val_results.speed.get("inference", 0)) if hasattr(val_results, "speed") else None,
        "fire_precision": per_class("fire", "p"),
        "fire_recall": per_class("fire", "r"),
        "fire_map50": per_class("fire", "ap50"),
        "smoke_precision": per_class("smoke", "p"),
        "smoke_recall": per_class("smoke", "r"),
        "smoke_map50": per_class("smoke", "ap50"),
    }


def _save_sample_predictions(model, val_images_dir, run_id, n=6):
    """Runs prediction on a few validation images and saves the annotated
    (boxes-drawn) output — the 'annotated clips' thumbnail strip."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    images = sorted(os.listdir(val_images_dir))[:n]
    saved = []
    for img_name in images:
        img_path = os.path.join(val_images_dir, img_name)
        try:
            result = model.predict(img_path, conf=0.25, verbose=False)[0]
            out_name = f"run{run_id}_{os.path.splitext(img_name)[0]}.jpg"
            result.save(filename=os.path.join(SAMPLES_DIR, out_name))
            saved.append(out_name)
        except Exception:
            continue
    return saved


def fine_tune(annotation_rows, run, db, epochs=20, start_epoch=0, resume_from=None, holdout_val_rows=None):
    """Generator: yields once per epoch so the caller (scheduler.py) can
    stop it if the 08:00 window boundary is reached mid-run. Writes metrics
    + sample predictions onto `run` (a TrainingRun row) as it goes, and
    commits via the passed-in `db` session — kept as parameters rather than
    imported here so this module has no Flask-app-context dependency of
    its own.

    start_epoch/resume_from let scheduler.py continue an interrupted run
    instead of restarting the full `epochs` count from BASE_CHECKPOINT —
    start_epoch is how many epochs the previous attempt already finished
    (run.epochs_completed), resume_from is the checkpoint it last saved
    (run.resume_checkpoint, refreshed every CHECKPOINT_EVERY epochs below).

    holdout_val_rows — annotated AnnotationImage rows with is_holdout=True
    (see config.VALIDATION_HOLDOUT_FOLDERS). When non-empty, _build_yolo_dataset
    uses them as the entire val split and puts every row of annotation_rows
    into train instead of carving a val split out of annotation_rows by index."""
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="nightly_train_") as tmp:
        data_yaml, val_images_dir = _build_yolo_dataset(annotation_rows, tmp, holdout_rows=holdout_val_rows)

        if resume_from and os.path.exists(resume_from):
            model = YOLO(resume_from)
        else:
            model = YOLO(BASE_CHECKPOINT if os.path.exists(BASE_CHECKPOINT) else "yolo11n.pt")

        # epochs=1 per call lets the scheduler check the window boundary
        # between epochs instead of only after the whole run finishes.
        #
        # batch=8 (down from ultralytics' batch=16 default): four straight
        # nightly runs on this board died mid-epoch with the Jetson
        # hard-rebooting; the last one showed RAM 94% + SWAP 100% full and
        # Xorg logging "your system is too slow" right before it died — this
        # is a 7.5GB unified-memory board, not enough headroom for the
        # default batch count against a ~8k-image dataset.
        #
        # workers=0 (down from an earlier workers=2): each `model.train()`
        # call in this loop builds a brand-new Trainer/DataLoader, and its
        # worker subprocesses were not being torn down before the *next*
        # call's workers spawned — confirmed directly (ps -o pid,ppid showed
        # 2 leftover workers from a finished epoch still alive as *children*
        # of this process, then 4 more once the next epoch's post-train
        # validation started its own loader) — so the worker count climbed
        # every epoch instead of staying flat at 2, and RAM went with it.
        # workers=0 loads data in this same process instead of forking any —
        # slower per-batch (no parallel prefetch/decode) but nothing to leak.
        # Resuming should remember how good the best epoch *before* this
        # process started already was, not reset the comparison to -1 and
        # risk a mediocre post-resume epoch overwriting a genuinely better
        # pre-resume one.
        best_map50 = run.best_map50 if run.best_map50 is not None else -1.0

        for _epoch in range(start_epoch, epochs):
            train_metrics = model.train(data=data_yaml, epochs=1, imgsz=640, device=DEVICE,
                                          batch=8, workers=0,
                                          project=os.path.join(HOME_DIR, "dashboard", "training_runs"),
                                          name="nightly", exist_ok=True, verbose=False)
            run.epochs_completed = _epoch + 1

            # train() already ran this epoch's validation internally to pick
            # its own (single-epoch-scoped) best.pt — reuse that result
            # instead of paying for a second validation pass.
            try:
                epoch_map50 = float(train_metrics.box.map50)
            except Exception:
                epoch_map50 = None
            if epoch_map50 is not None and epoch_map50 > best_map50:
                best_map50 = epoch_map50
                run.best_map50 = best_map50
                last = os.path.join(HOME_DIR, "dashboard", "training_runs", "nightly", "weights", "last.pt")
                if os.path.exists(last):
                    shutil.copy2(last, TRUE_BEST_CHECKPOINT_PATH)

            if run.epochs_completed % CHECKPOINT_EVERY == 0:
                last = os.path.join(HOME_DIR, "dashboard", "training_runs", "nightly", "weights", "last.pt")
                if os.path.exists(last):
                    shutil.copy2(last, RESUME_CHECKPOINT_PATH)
                    run.resume_checkpoint = RESUME_CHECKPOINT_PATH

            db.session.commit()
            yield _epoch

        # Deploy whichever epoch was actually best, not just whichever ran
        # last — falls back to the old (last-epoch) behavior only if metrics
        # extraction failed for every single epoch above. Checking
        # run.best_map50 (not just the file's existence) matters here: if
        # this run's own metric extraction failed every epoch, best_map50
        # stays None even though TRUE_BEST_CHECKPOINT_PATH might still exist
        # on disk as a stale leftover from a *previous, unrelated* run —
        # without this check that old file would get silently deployed.
        if run.best_map50 is not None and os.path.exists(TRUE_BEST_CHECKPOINT_PATH):
            model = YOLO(TRUE_BEST_CHECKPOINT_PATH)
            shutil.copy2(TRUE_BEST_CHECKPOINT_PATH, BASE_CHECKPOINT)
        else:
            best = os.path.join(HOME_DIR, "dashboard", "training_runs", "nightly", "weights", "best.pt")
            if os.path.exists(best):
                model = YOLO(best)
                shutil.copy2(best, BASE_CHECKPOINT)

        val_results = model.val(data=data_yaml, device=DEVICE, workers=0, verbose=False)
        metrics = _extract_metrics(val_results, model)
        for key, value in metrics.items():
            setattr(run, key, value)

        samples = _save_sample_predictions(model, val_images_dir, run.id)
        run.sample_images = ",".join(samples)

        # Run finished all its epochs normally — the resume checkpoint has
        # served its purpose (BASE_CHECKPOINT now holds the final weights).
        if run.resume_checkpoint and os.path.exists(run.resume_checkpoint):
            try:
                os.remove(run.resume_checkpoint)
            except OSError:
                pass
        run.resume_checkpoint = None
        db.session.commit()
