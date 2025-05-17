# Transformer-Anomaly-Detection-Surveillance
Transformer-based anomaly detection framework for surveillance videos using segment-level Multiple Instance Learning MIL

---

## 🔍 Overview

- ⏱ Real-time inference on video segments
- 🧠 Feature extraction with TimeSformer (CLS token)
- 📦 MIL-based anomaly scoring
- 📊 Segment-level ROC-AUC evaluation with manual annotations

---

## 📊 Methodology

### Step 1: Segment-Level Feature Extraction

Each video is divided into **non-overlapping segments**. For each segment (16 frames), we extract a 768-dimensional [CLS] token feature using the **TimeSformer** model.

![Preprocessing Pipeline](preprocessing.jpg)

> 🔹 This step converts a video into a bag of temporal embeddings:  
> `Video = [segment_0_feat, segment_1_feat, ..., segment_N_feat]`  

---

### Step 2: Multiple Instance Learning (MIL)

The extracted segment features are passed through a lightweight **MIL classifier** that assigns an anomaly score to each segment. Only a weak video-level label is used during training.

![Model Architecture](system_architecture.jpg)

---


### Step 4: Evaluation

Each segment is compared against **manually annotated anomaly intervals** using ROC-AUC. This ensures fine-grained, interpretable performance.

![ROC-AUC Curve](results/roc.PNG)

![Anomaly Scores](results/anomalyscores.jpg)


---

## 🚀 Quick Start

```bash
git clone https://github.com/umarzaib123/Transformer-Anomaly-Detection-Surveillance.git
cd Transformer-Anomaly-Detection-Surveillance
pip install -r requirements.txt

