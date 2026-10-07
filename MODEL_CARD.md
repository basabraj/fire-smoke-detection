---
pipeline_tag: object-detection
tags:
- fire
- smoke
- safety
- pytorch
base_model:
- Ultralytics/YOLO26
---

# Safety Detection

A fine-tuned YOLO model for detecting fire and smoke in images and video streams, built for real-time safety monitoring.

## Model Details
- **Architecture:** YOLOv26 (fine-tuned)
- **Framework:** PyTorch
- **Epochs:** 52
- **Experiment Tracking:** ClearML

## Classes
| ID | Label |
|----|-------|
| 0  | fire  |
| 1  | smoke |

## Dataset
Fine-tuned on the [Home Fire Dataset](https://www.kaggle.com/datasets/pengbo00/home-fire-dataset) from Kaggle.


- **Training Logs:** [ClearML Experiment](https://app.clear.ml/projects/bffe65b5fe1649dd9d202e181ba92fe0/tasks/f57871573c9d4d969dd5867004857d99/scalars)


## Evaluation Metrics
| Metric    | Value |
|-----------|-------|
| mAP@50    | 0.930 |
| mAP@50-95 | 0.626 |
| Precision | 0.913 |
| Recall    | 0.891 |

## Usage
```python
from ultralytics import YOLO

model = YOLO("path/to/model.pt")
results = model("image.jpg")
```

## Limitations
- Trained on home fire scenarios — performance may degrade in industrial or outdoor environments
- Detection confidence decreases at stricter IoU thresholds (mAP@50-95: 0.626)
- Not validated for production safety-critical systems without further testing