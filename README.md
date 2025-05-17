# Transformer-Anomaly-Detection-Surveillance
Transformer-based anomaly detection framework for surveillance videos using segment-level Multiple Instance Learning MIL

---

## 🔍 Overview

- ⏱ Real-time inference on video segments
- 🧠 Feature extraction with TimeSformer (CLS token)
- 📦 MIL-based anomaly scoring
- 📊 Segment-level ROC-AUC evaluation with manual annotations

---

## 🗂 Project Structure

- `custom_dataset/` custom videos created
- `models/` — Trained MIL models
- `results/` — AUC/ROC visualizations

---

## 🧪 Preprocessing

Each video is split into non-overlapping segments of 16 frames. TimeSformer is used to extract a 768-D CLS feature vector per segment.

---

## 🚀 Quick Start

```bash
git clone https://github.com/umarzaib123/Transformer-Anomaly-Detection-Surveillance.git
cd Transformer-Anomaly-Detection-Surveillance
pip install -r requirements.txt

