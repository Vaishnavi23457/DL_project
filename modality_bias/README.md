# Modality Bias — XAI-Guided Detection and Mitigation

Part of the project: **Development of Methods to Address Modality Bias in Multimodal Deep Learning Models**

This folder contains the modality bias analysis and proposed mitigation method,
built on top of the baseline OPM/OGM-GE implementation in `../balanced_mm/`.

---

## Research Questions

**RQ1:** Can XAI attribution scores be used to identify and mitigate modality bias?  
**RQ2:** Can a novel adaptive attention mechanism learn to balance modality contributions?

---

## Pipeline

```
Multimodal Model (train baseline)
        ↓
Modality Bias Score (MBS + rho)
        ↓
Integrated Gradients → XAI Scores
        ↓
Pearson + Spearman: XAI ↔ MBS        ← RQ1 Part A
        ↓
Modality Ablation → Confidence Drop
        ↓
XAI ↔ Ablation Reliance               ← RQ1 Part B
        ↓
Layer-wise Bias Analysis
        ↓
OPM Mitigation (from balanced_mm/)
        ↓
XAI-Guided Adaptive Attention + OPM   ← RQ2 (Proposed)
        ↓
Final Comparison: Baseline vs OPM vs Proposed
```

---

## Datasets

| Dataset    | Modalities          | Task           |
|------------|---------------------|----------------|
| CREMA-D    | Audio + Video       | Emotion (6cls) |
| Food-101   | Image + Text        | Food (101cls)  |
| CMU-MOSEI  | Audio + Video + Text| Emotion (6cls) |

---

## Project Structure

```
modality_bias/
├── mb/                        ← Python modules
│   ├── cremad_data.py         ← CREMA-D dataset loader
│   ├── cmu_mosei_data.py      ← CMU-MOSEI dataset loader
│   ├── ig_xai_av.py           ← Integrated Gradients for audio+video
│   ├── mbs.py                 ← Modality Bias Score
│   ├── correlation.py         ← XAI ↔ MBS correlation
│   ├── ablation.py            ← Modality ablation + confidence drop
│   └── layerwise.py           ← Per-layer bias analysis
│
├── notebooks/
│   ├── 01_setup_and_model.ipynb        ← Data setup + baseline training
│   ├── 02_bias_and_xai.ipynb           ← MBS + XAI + correlation + ablation + layerwise
│   ├── 03_mitigation_and_proposed.ipynb← OPM + XAI-guided proposed method
│   └── 04_comparison_report.ipynb      ← Final comparison + visualizations
│
├── results/                   ← JSON results + figures (committed to GitHub)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/DL_project.git
cd DL_project/modality_bias
pip install -r requirements.txt
```

### 2. Add to Python path (in notebooks)
```python
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # for balanced_mm/bml/
sys.path.insert(0, os.path.abspath("."))            # for mb/
```

### 3. Download datasets
- **CREMA-D**: [Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad) or [GitHub](https://github.com/CheyneyComputerScience/CREMA-D)
- **CMU-MOSEI**: [CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)
- **Food-101**: handled by `../balanced_mm/` (see its README)

---

## Running on Colab / Kaggle

Each notebook starts with:
```python
# Clone repo
!git clone https://github.com/<your-username>/DL_project.git
%cd DL_project/modality_bias
!pip install -q -r requirements.txt

# Mount Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')
SHARED = "/content/drive/MyDrive/modality_bias_shared"
```

Large files (checkpoints, processed data) are stored in the shared Drive folder.
Results JSONs and figures are committed to this GitHub repo.

---


