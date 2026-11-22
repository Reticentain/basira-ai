# Basira AI (بصیرت)
### A Watchful Eye for Every Clinic
**Offline, Explainable AI Screening for Diabetic Retinopathy in Low-Resource Settings**

Air University, Islamabad — NCSA | Artificial Intelligence, Fall 2026

| Team Member | Roll No. |
|---|---|
| Areeza Batool | 242367 |
| Hammad Farid | 242355 |
| Ain Anshara | 242303 |

**Instructor:** Bilal Saleem &nbsp;·&nbsp; **Supervisor:** NCSA, Air University

> **Scope:** This is a university course project. Basira AI is decision support,
> not a medical device. All results are from retrospective public data and must not
> be used to make clinical decisions without prospective validation.

---

## What is Basira AI?

Basira AI takes a retinal fundus photograph and returns:
- a **quality verdict** (usable or "please retake"),
- a **DR grade** from 0 to 4 (secondary task),
- a **referral decision** — grade ≥ 2 means refer to a specialist (primary task),
- a **calibrated confidence score**, and
- a **Grad-CAM heatmap** showing which retinal regions drove the prediction.

The system runs **offline on modest hardware**, with English and Urdu output.

---

## Repository Structure

```
basira-ai/
├── configs/
│   ├── default.yaml            ← all paths, image size, seeds, Drive output paths
│   └── local.yaml.example      ← copy → configs/local.yaml for local overrides (git-ignored)
├── src/
│   └── basira/
│       ├── config.py           ← load_config() — single config entry point
│       ├── data/
│       │   └── splits.py       ← patient-level split + assert_no_patient_leakage()
│       ├── models/             ← (Phase 2) EfficientNet-B0, MobileNetV3
│       ├── evaluation/         ← (Phase 2) AUC, kappa, calibration, CI
│       └── explainability/     ← (Phase 3) Grad-CAM overlays
├── notebooks/
│   └── 00_colab_bootstrap.ipynb  ← run this first every Colab session
├── scripts/                    ← (Phase 2+) train.py, evaluate.py, export.py
├── app/                        ← (Phase 4) offline web / Android app
├── tests/
│   └── test_patient_split.py   ← patient-leakage guard (must pass before any run)
├── reports/                    ← generated JSON/CSV/figures from evaluation scripts
├── data/
│   └── README.md               ← dataset download instructions
├── requirements-colab.txt      ← pip deps for Colab (no torch/torchvision)
└── README.md
```

> **Data is never committed.** `data/`, `*.pt`, `*.onnx`, API tokens are all git-ignored.

---

## Running Everything in Google Colab

**All training, evaluation, and data processing runs in Colab — not on your local machine.**

### One-time setup

1. **Configure Kaggle API credentials** in Colab (or use the Kaggle API in a cell) to download datasets. Never commit credentials.
2. **Mount Google Drive** so checkpoints and reports survive session resets:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. All persistent outputs go to `/content/drive/MyDrive/basira/` (set in `configs/default.yaml`).

### Every new Colab session — run this once:

Open **`notebooks/00_colab_bootstrap.ipynb`** and run all cells. It will:

| Step | What it does |
|---|---|
| 1 | Mount Google Drive |
| 2 | Clone or `git pull` the repo into `/content/basira-ai/` |
| 3 | `pip install -q -r requirements-colab.txt` |
| 4 | Check GPU (warns if no GPU found) |
| 5 | Load config and set global seeds |
| 6 | Count images per dataset and report any missing folders |
| 7 | Run `pytest tests/test_patient_split.py` |

### Notebook order

| Notebook | Purpose | Status |
|---|---|---|
| `00_colab_bootstrap.ipynb` | Session setup | ✅ Done |
| `01_eda.ipynb` | Exploratory data analysis, image-size measurement | Phase 2 |
| `02_preprocessing.ipynb` | Crop, CLAHE, splits | Phase 2 |
| `03_training.ipynb` | Model training with checkpointing | Phase 2 |
| `04_evaluation.ipynb` | AUC, kappa, calibration, CI, cross-dataset | Phase 3 |
| `05_gradcam.ipynb` | Grad-CAM overlays and failure analysis | Phase 3 |
| `06_compression.ipynb` | int8 quantisation, ONNX export | Phase 4 |

---

## Datasets

| Dataset | Role | Size |
|---|---|---|
| EyePACS (Kaggle 2015 resized) | Training + internal val | 35 k train + 53 k test JPEGs |
| APTOS 2019 | Training + internal val/test | ~3 700 PNGs |
| Messidor-2 | **External test only** | ~1 748 images |
| IDRiD | **External test only** | 516 images |

See [`data/README.md`](data/README.md) for exact download commands and access-request links.

---

## Core Rules (enforced in code)

| Rule | Where enforced |
|---|---|
| Splits are **by patient** (EyePACS) or by image with deduplication (APTOS) | `src/basira/data/splits.py` + pytest |
| **No patient leaks** between train / val / test | `assert_no_patient_leakage()` called at split time |
| Thresholds tuned **only on the source validation set** | Never on Messidor-2, IDRiD, or test sets |
| **Never invent numbers** — every metric comes from a saved script output | `reports/*.json` or `reports/*.csv`; if not run → "not run yet" |
| **Bootstrap CI by patient** (EyePACS) — both eyes of one person are correlated | `src/basira/evaluation/` (Phase 2) |
| All paths from `configs/default.yaml` — **nothing hard-coded** | `load_config()` enforces this |
| **No data or tokens committed** | `.gitignore` |

---

## Tasks and Labels

| Task | Type | Positive class |
|---|---|---|
| **Primary:** referable DR | Binary (grade ≥ 2) | Referable |
| **Secondary:** DR grading | 5-class (0 = None → 4 = Proliferative) | — |

---

## Models

| Model | Purpose | Library |
|---|---|---|
| **EfficientNet-B0** | Main model | `timm` (ImageNet pretrained) |
| MobileNetV3-Large | Lightweight comparison | `timm` (ImageNet pretrained) |

---

## Metrics (all with 95% CI by bootstrap)

| Metric | Task |
|---|---|
| AUC (ROC) | Referable binary |
| Sensitivity at fixed specificity (target: 90%) | Referable binary |
| Quadratic weighted kappa | 5-class grading |
| Confusion matrix | Both |
| Calibration (reliability diagram) | Referable binary |

---

## Evaluation Plan

| Experiment | Datasets | Status |
|---|---|---|
| In-domain — EyePACS | EyePACS test split | Not run yet |
| In-domain — APTOS | APTOS test split | Not run yet |
| External — Messidor-2 | Messidor-2 (never trained on) | Not run yet |
| External — IDRiD | IDRiD (never trained on) | Not run yet |
| Robustness sweep | Test sets + synthetic perturbations | Not run yet |
| Compression (fp32 vs int8) | EyePACS test split | Not run yet |

---

## Environment

| Component | Library / Tool |
|---|---|
| Deep learning | PyTorch + `timm` |
| Config management | `omegaconf` |
| Image processing | OpenCV (headless), Pillow |
| Evaluation | scikit-learn, scipy |
| Explainability | `pytorch-grad-cam` |
| Export | `onnx`, `onnxruntime` |
| Testing | `pytest` |
| Training compute | Google Colab / Kaggle (free GPU) |

---

## Privacy and Ethics

- No patient data is committed or uploaded to any cloud AI service.
- Basira AI is decision support only — not a diagnostic device.
- Track B (hospital data) requires ethics (IRB) approval before any use.
- All results are from retrospective public data; prospective clinical validation is future work.
