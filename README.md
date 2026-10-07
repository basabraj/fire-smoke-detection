<img width="1600" height="805" alt="front" src="https://github.com/user-attachments/assets/b4f2b6f6-0c4a-4f90-b57e-ac7d5aad7768" />

<img width="1600" height="806" alt="Overview" src="https://github.com/user-attachments/assets/f26b527e-ccdb-4a32-b17e-2357296ac2fb" />

<img width="1600" height="806" alt="Alert" src="https://github.com/user-attachments/assets/49320f83-2aad-4288-8a45-8923f72db944" />

<img width="1600" height="806" alt="training1" src="https://github.com/user-attachments/assets/8bf098e3-3286-4662-b001-67c3928e2496" />

<img width="1600" height="807" alt="training2" src="https://github.com/user-attachments/assets/f8ad6558-5191-4b22-b357-f8691d58f943" />

<img width="1600" height="810" alt="training3" src="https://github.com/user-attachments/assets/ff81517d-aa14-434b-9308-007b89e6d0f5" />

<img width="1917" height="971" alt="training4" src="https://github.com/user-attachments/assets/40dd8649-f4f9-4e5f-ab59-9ad64da9ca5a" />

<img width="1917" height="968" alt="trained_images" src="https://github.com/user-attachments/assets/bae8cf48-77ef-4c43-b351-d5307e993c05" />

<img width="1830" height="822" alt="training_history" src="https://github.com/user-attachments/assets/e695dc1e-dc0e-4a8e-a17e-4266892ab6f8" />

<img width="1897" height="910" alt="training_metrics" src="https://github.com/user-attachments/assets/ca4c9059-109c-4869-be85-a68215264b03" />


<img width="1600" height="801" alt="settings" src="https://github.com/user-attachments/assets/2d4ed013-2bbb-485b-9460-88635ad4d3f6" />




# Fire & Smoke Detection (Jetson Orin Nano)

Edge fire and smoke detection for CCTV. A YOLO model runs on an RTSP camera stream, detections are logged to a Flask + SQLite dashboard with a live feed, and the model is retrained on a schedule from images you annotate in the browser.

Built and run on an NVIDIA Jetson Orin Nano (JetPack, CUDA). It will also run on CPU, just slowly.

## Features

- **Live detection** on an RTSP stream (`dashboard/camera_worker.py`, GStreamer + OpenCV), with a live feed at `/video_feed`.
- **Dashboard** with today's stats, weekly/monthly charts, a fire-vs-smoke split and a detection table with saved frames.
- **In-browser annotation** (Training page): draw boxes, mark "no fire / no smoke / no detection", review flagged false positives.
- **Trained Images page**: browse what has been trained on, what is annotated but waiting, and what is still pending. AI-labeled boxes are marked so they can be reviewed.
- **Scheduled retraining** (`dashboard/scheduler.py`, `dashboard/train_job.py`) with crash-safe resume and best-epoch selection.
- **Held-out validation set** that is never used for training.
- **Alert page** with a phone number, cooldown and history. SMS sending is a stub (see below).

## Layout

```
Home/
├── yolo_smoke_fire.pt        # deployed model (classes: 0 = fire, 1 = smoke)
├── MODEL_CARD.md             # model card (metrics, dataset, limitations)
├── rtsp_stream.py            # standalone RTSP detector script
├── jetson_cam.py, picam_stream.py, gst_test_fixed.py   # camera helpers
├── extract_frames.py         # video -> 1 fps frames
├── dashboard/
│   ├── app.py                # Flask app and all routes (port 5050)
│   ├── camera_worker.py      # live inference thread
│   ├── scheduler.py          # hourly image scan + scheduled training
│   ├── train_job.py          # fine-tuning, checkpoints, best-epoch deploy
│   ├── manual_train_now.py   # run training immediately from the terminal
│   ├── sms_alert.py          # alert message + cooldown (provider not wired)
│   ├── models.py, config.py
│   ├── templates/, static/
│   └── requirements.txt
└── dataset/fire_smoke/       # fire/ smoke/ no_fire/ no_smoke/ validation_holdout/ flagged_review/
```

## Setup

```bash
python3 -m venv fsenv
source fsenv/bin/activate
pip install -r dashboard/requirements.txt
```

On a Jetson, install the NVIDIA-provided PyTorch build for your JetPack version first, and make sure GStreamer with its Python bindings is available. The dashboard reads the camera through GStreamer.

Run it:

```bash
cd dashboard
python app.py          # http://<device-ip>:5050
```

To keep it running across reboots, use a systemd user service pointing at `dashboard/app.py` and enable lingering for your user.

The camera RTSP URL and the fire/smoke confidence thresholds (default 0.7 each) are set on the **Settings** page. They are stored in `dashboard/dashboard.db`, not in code.

## Training pipeline

1. Put frames in `dataset/fire_smoke/{fire,smoke}/frames/`. The hourly scanner registers new images.
2. Annotate them on the Training page. Labels are saved as YOLO `.txt` files next to each image.
3. `no_fire/frames/` and `no_smoke/frames/` need no annotation; they are registered as ready-to-train negatives.
4. The scheduled run starts at 14:00, Monday to Saturday (`config.py`), and stops at the end of the window. It checkpoints every 5 epochs and resumes after a crash or window cutoff.
5. The best epoch by mAP50 is deployed to `yolo_smoke_fire.pt`, not just the last one.

Run training immediately instead of waiting for the window:

```bash
cd dashboard
python manual_train_now.py
```

### Validation set

Frames of the same video are near-duplicates, so a random or index split lets the validation set "see" training data and inflates scores. Instead, put separate videos' frames in `dataset/fire_smoke/validation_holdout/{fire,smoke}/frames/`. They are registered with `is_holdout=True`, annotated on the Training page's **Validation Set** tab, used only for validation, and excluded from the training pool. Until any are annotated, training falls back to a 90/10 split.

Note: if you pre-label validation images with the model itself, review them before trusting the score, otherwise the model is being graded on its own predictions.

## Security notes

- Camera credentials live in `dashboard/dashboard.db`. Do not commit the database.
- A few helper scripts (`rtsp_stream.py`, `jetson_cam.py`, `gst_test_fixed.py`) contain a hardcoded `RTSP_URL`. Replace it with an environment variable or the Settings-page value before publishing this repository.
- The SMS provider key is read from the `SMS_PROVIDER_API_KEY` environment variable. Never put it in code.
- `config.py` ships a development `SECRET_KEY`. Change it for anything exposed beyond a trusted network.
- The Flask development server is used. Put it behind a reverse proxy if it is reachable from outside your LAN.

## SMS alerts

`dashboard/sms_alert.py` builds the message, enforces a cooldown and records history, but no SMS gateway is connected yet. Until you fill in the provider call in `_dispatch()`, alerts are logged as "simulated".

## Model

See [MODEL_CARD.md](MODEL_CARD.md) for the base model, dataset, and limitations. Not validated for use as a sole safety system.
