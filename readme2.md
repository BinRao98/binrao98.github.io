# TruckAct: A Multi-source, Multimodal Dataset for Real-world Truck Driver Activity Recognition

<p align="center">
  <img src="images/framework.png" width="90%">
</p>

# Introduction

This repository provides the official implementation of **TruckAct**, a
multi-source and multimodal benchmark for **real-world truck driver activity
recognition**. TruckAct introduces a naturalistic driving dataset collected from professional
truck drivers during real-world long-haul transportation operations. The
dataset provides synchronized multi-modal observations, including in-cabin RGB
videos, vehicle telemetry, and wearable wristband signals, enabling research on
robust driver activity understanding under realistic driving conditions. In addition to the dataset, this repository provides **TruckAct-Net** as a
baseline model for multimodal driver activity recognition. The baseline
integrates visual and non-visual information through a multi-stream architecture
with adaptive multimodal fusion, providing a reproducible benchmark for future
research.

---

# Table of Contents

- [News](#news)
- [Dataset Overview](#dataset-overview)
  - [Dataset Access](#dataset-access)
  - [Dataset Statistics](#dataset-statistics)
  - [Modalities](#modalities)
  - [Activity Categories](#activity-categories)
- [Framework Overview](#framework-overview)
- [Performance](#performance)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)


# News

- **2026/08/23**: TruckAct paper released and the benchmark code is publicly available.
- **2026/08/24**: The TruckAct dataset is available through restricted access.

---

# Dataset Overview

TruckAct is a naturalistic multimodal dataset dedicated to truck driver
activity recognition. Unlike existing datasets mainly collected from passenger cars or controlled
environments, TruckAct captures spontaneous driver behaviors during real-world
commercial trucking operations.

## Dataset Access

Because the dataset contains human-subject driving recordings, the image
data have been mosaicked to protect driver privacy. The dataset files are
distributed through Zenodo under restricted access.

To request access:

1. Complete the [data-use questionnaire](https://forms.gle/B4BKVGyw6HyGLWw27).
2. Submit an access request through the
   [Zenodo dataset record](xxx-xxx-xxx).
3. Access will be granted after the request has been reviewed.

The questionnaire collects the applicant's email address, name, institution,
and intended use of the data. Users must agree to:

- not redistribute the dataset;
- not attempt participant identification;
- use the dataset only for approved research purposes.

---

## Dataset Statistics

| Property | Description |
|---|---|
| Drivers | 9 professional truck drivers |
| Driving trips | 62 long-haul trips |
| Total duration | Approximately 67 hours |
| Data samples | 24,000 non-overlapping 10-second clips |
| Driver activities | 9 activity classes |
| Modalities | RGB video + vehicle telemetry + wristband signals |

All recordings were collected without scripted behaviors or artificial
intervention.

---

## Modalities

TruckAct contains three synchronized input modalities:

### 1. In-cabin RGB Video

- Captured from windshield-mounted cameras
- Resolution: 1280 × 720
- Frame rate: 25 FPS
- Used to capture driver posture, hand movements, and object interactions

### 2. Vehicle Telemetry

Vehicle signals include:

- Vehicle speed
- Tri-axis acceleration
- GPS information
- Elevation
- Cabin temperature
- ...

### 3. Wristband Signals

The wearable wristband records:

- Heart rate (HR)
- Galvanic skin response (GSR)
- Photoplethysmography (PPG)
- Tri-axis wrist motion
- ...

Missing wristband observations are preserved to simulate realistic deployment
conditions.

---

## Activity Categories

TruckAct contains nine mutually exclusive driver activity classes:

| Class | Activity |
|---|---|
| 0 | Normal driving |
| 1 | Handheld calling |
| 2 | Phone interaction |
| 3 | Dashboard operation |
| 4 | Eating / drinking |
| 5 | Reaching behind |
| 6 | Grooming / adjusting |
| 7 | Headset calling |
| 8 | Smoking |

---

# Framework Overview

To establish a reproducible benchmark, we provide **TruckAct-Net** as the
baseline model for TruckAct.

TruckAct-Net adopts a multi-stream multimodal architecture that processes
different sensing sources independently and combines complementary information
for activity recognition.

The baseline framework consists of:

- **Visual stream:** extracts spatio-temporal representations from RGB video
  clips.
- **Sensor streams:** encode sequential vehicle telemetry and wristband signals
  to capture driving dynamics and physiological variations.
- **Multimodal fusion:** combines representations from available modalities with
  adaptive weighting to improve robustness under sensor noise or missing data.

The baseline supports different modality configurations, including:

- RGB video only;
- RGB + vehicle telemetry;
- RGB + wristband signals;
- RGB + vehicle telemetry + wristband signals.

This implementation serves as a reference benchmark, and researchers are
encouraged to develop new methods based on the TruckAct dataset.

---

# Performance

TruckAct-Net provides a reference performance on the TruckAct benchmark.

## Trimodal Setting

| Model | Accuracy | Macro Recall | Macro Precision | Macro F1 |
|---|---:|---:|---:|---:|
| TruckAct-Net | **77.34%** | **49.16%** | **51.50%** | **46.10%** |

The results are obtained using synchronized RGB video, vehicle telemetry, and
wristband signals.

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch environment is required.

---

# Data Preparation

The expected dataset structure is:

```
data/
├── picture_data3/
│   └── <vehicle_id>/
│       └── <vehicle_id>_<trip_id>/
│           └── <video_id>_<10s>/
│               ├── 001.png
│               ├── 002.png
│               └── ...
│
├── processed_sensor_data/
│   ├── vehicle_processed.npz
│   └── wristband_processed.npz
│
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Each split CSV should contain:

```
车辆编号, 行程编号, 视频编号, 10s, Tag, segment_id
```

Processed sensor files should contain:

```
segment_id
features
mask
length
```

---

# Training

Example:

```bash
python train.py custom RGB \
  --train_csv /data/splits/train.csv \
  --val_csv /data/splits/val.csv \
  --img_root /data/picture_data3 \
  --veh_npz /data/processed_sensor_data/vehicle_processed.npz \
  --phy_npz /data/processed_sensor_data/wristband_processed.npz \
  --tune_from /checkpoints/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
  --pretrain none \
  --arch resnet50 \
  --num_segments 16 \
  --mmode E \
  --fusion_mode adaptive_logit \
  --sequence_encoder xformer \
  --batch-size 8 \
  --workers 4 \
  --epochs 50
```

---

# Testing

Example:

```bash
python test.py custom RGB \
  --test_csv /data/splits/test.csv \
  --img_root /data/picture_data3 \
  --veh_npz /data/processed_sensor_data/vehicle_processed.npz \
  --phy_npz /data/processed_sensor_data/wristband_processed.npz \
  --resume /runs/truckact_e/best.pth \
  --tune_from /checkpoints/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
  --pretrain none \
  --arch resnet50 \
  --num_segments 16 \
  --mmode E \
  --fusion_mode adaptive_logit \
  --sequence_encoder xformer \
  --batch-size 8 \
  --workers 4
```

The evaluation script reports:

- Top-1 accuracy
- Top-5 accuracy
- Mean-class accuracy
- ...

---

# Citation

If you use TruckAct in your research, please cite:

```bibtex
@article{wang2026truckact,
  title={TruckAct: A multi-source, multimodal dataset for real-world truck driver activity recognition},
  author={Wang, Qianfang and Rao, Bin and Pei, Xin and Xu, Pengpeng and Chen, Tiantian},
  journal={},
  year={2026}
}
```


---

# Acknowledgement

This work was supported by:

- National Natural Science Foundation of China
- Natural Science Foundation of Guangdong Province, China

We thank all participating truck drivers and collaborators for supporting the
naturalistic data collection.

---
