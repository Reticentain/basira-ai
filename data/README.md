# Data Download Guide — Basira AI

> **Privacy rule:** image data is **never committed to this repository** and is listed in `.gitignore`.
> All paths are configured in `configs/default.yaml`; override locally via `configs/local.yaml`.

---

## Target directory layout (Colab / Kaggle default: `/content/data`)

```
/content/data/
├── aptos/
│   ├── train_images/        ← *.png  (~3 662 images)
│   └── train.csv            ← columns: id_code, diagnosis (0-4)
├── eyepacs/
│   ├── resized train 15/    ← *.jpg  (35 126 images)
│   ├── resized test 15/     ← *.jpg  (53 576 images)
│   └── labels/
│       ├── trainLabels15.csv   ← columns: image, level
│       └── testLabels15.csv    ← columns: image, level, Usage
├── messidor2/
│   ├── images/              ← *.jpg / *.png
│   └── labels.csv           ← columns: image, grade
└── idrid/
    ├── images/              ← *.jpg  (all 516 images)
    └── labels.csv           ← columns: image, grade, orig_split
```

---

## Dataset 1 — APTOS 2019

| Property | Value |
|---|---|
| Full name | APTOS 2019 Blindness Detection |
| Origin | Asia Pacific Tele-Ophthalmology Society, India |
| Images | ~3 662 labelled PNG fundus photos |
| Labels | `id_code`, `diagnosis` (0 = No DR, 4 = Proliferative DR) |
| Patient IDs | **Not provided** — treat each image as a separate patient |
| Role in Basira AI | Training + internal validation (patient-level split) |
| Licence | Kaggle competition terms (non-commercial research) |

### How to download

1. Accept the Kaggle competition terms at  
   <https://www.kaggle.com/competitions/aptos2019-blindness-detection/data>
2. Using the Kaggle CLI (already installed on Colab):
   ```bash
   pip install -q kaggle
   # Place your Kaggle API token in ~/.kaggle/ first, then:
   mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/*

   kaggle competitions download -c aptos2019-blindness-detection -p /content/data/aptos
   cd /content/data/aptos && unzip -q aptos2019-blindness-detection.zip
   # Rename the extracted folder so train images land in train_images/
   mv train_images /content/data/aptos/train_images
   mv train.csv    /content/data/aptos/train.csv
   ```
3. Verify:
   ```bash
   ls /content/data/aptos/train_images | head -5
   head -3 /content/data/aptos/train.csv
   ```

---

## Dataset 2 — EyePACS (Kaggle 2015 resized mirror)

| Property | Value |
|---|---|
| Full name | Diabetic Retinopathy Detection (Kaggle 2015) — resized 15 mirror |
| Origin | EyePACS / California Healthcare Foundation screening programme, USA |
| Images | 35 126 train + 53 576 test JPEG fundus photos |
| Labels | `image`, `level` (0–4); test CSV also has `Usage` column |
| Patient IDs | Encoded in filename: `<patient_id>_<left|right>.jpg` |
| Role in Basira AI | Main large-scale training source; train/val split by patient |
| Official train/test | No patient overlap between the two official sets |
| Licence | Kaggle competition terms (non-commercial research) |

### How to download

The competition originally required a local download; a resized mirror is on Kaggle Datasets:

```bash
# Mirror dataset (Ben Graham's resized version — most commonly cited)
kaggle datasets download -d benjaminwarner/resized-2015-2019-blindness-detection-images \
      -p /content/data/eyepacs --unzip
# Or use the direct competition download (original full size, ~35 GB):
# kaggle competitions download -c diabetic-retinopathy-detection -p /content/data/eyepacs
```

After extraction, ensure the directory names match the config exactly:
```
/content/data/eyepacs/resized train 15/   (note the spaces)
/content/data/eyepacs/resized test 15/
/content/data/eyepacs/labels/trainLabels15.csv
/content/data/eyepacs/labels/testLabels15.csv
```

If the extracted names differ, rename them or update `configs/default.yaml`.

> **Access note:** You must have accepted the [Kaggle competition rules](https://www.kaggle.com/c/diabetic-retinopathy-detection) before the API will serve the data.

---

## Dataset 3 — Messidor-2 (external test only)

| Property | Value |
|---|---|
| Full name | Messidor-2 |
| Origin | French hospital screening programme |
| Images | ~1 748 fundus photos (two eyes per patient) |
| Labels | Adjudicated DR grade 0–4 from Abràmoff et al. 2016 |
| Role in Basira AI | **External test only** — never used for training or tuning |
| Licence | Free for non-commercial research; registration required |

### How to download

Messidor-2 requires a **data-access request**:

1. Go to <https://www.adcis.net/en/third-party/messidor2/> and click **"Download"**.
2. Fill in the registration form (name, institution, intended use).  
   An e-mail with a download link is typically sent within a few hours to a couple of days.
3. Download and extract; place images and the adjudicated labels CSV at:
   ```
   /content/data/messidor2/images/
   /content/data/messidor2/labels.csv    # columns: image, grade
   ```
4. The adjudicated labels (Abràmoff 2016 grading) are distributed separately; link is on the same page.

> **No automated download is possible** — a manual e-mail registration is mandatory.

---

## Dataset 4 — IDRiD (external test only)

| Property | Value |
|---|---|
| Full name | Indian Diabetic Retinopathy Image Dataset |
| Origin | Shri Guru Gobind Singhji Eye Hospital, Nanded, India |
| Images | 516 fundus photos with per-image DR grade and lesion annotations |
| Labels | DR grade 0–4 (`grade`), original split designation (`orig_split`) |
| Role in Basira AI | **External test only** — all 516 images used, no further splitting |
| Licence | Creative Commons — free for research; registration required |

### How to download

IDRiD is hosted on the IEEE DataPort and Grand Challenge:

**Option A — IEEE DataPort (requires free account):**
1. Create a free account at <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid>.
2. Log in and click the download links for the **Disease Grading** sub-task (not the segmentation sub-task).
3. Place files:
   ```
   /content/data/idrid/images/
   /content/data/idrid/labels.csv   # columns: image, grade, orig_split
   ```

**Option B — Grand Challenge archive:**
1. Go to <https://idrid.grand-challenge.org/>.
2. Register and request access.
3. Download and place as above.

> **Access note:** Both routes are free but require account creation and agreement to the dataset licence.

---

## Verifying your download

After placing all files, run the data integrity check:

```bash
cd /path/to/basira-ai
python - <<'EOF'
from pathlib import Path
import pandas as pd

checks = {
    "APTOS train_images": Path("/content/data/aptos/train_images"),
    "APTOS train.csv":    Path("/content/data/aptos/train.csv"),
    "EyePACS train":      Path("/content/data/eyepacs/resized train 15"),
    "EyePACS test":       Path("/content/data/eyepacs/resized test 15"),
    "EyePACS trainLabels":Path("/content/data/eyepacs/labels/trainLabels15.csv"),
    "EyePACS testLabels": Path("/content/data/eyepacs/labels/testLabels15.csv"),
    "Messidor-2 images":  Path("/content/data/messidor2/images"),
    "Messidor-2 labels":  Path("/content/data/messidor2/labels.csv"),
    "IDRiD images":       Path("/content/data/idrid/images"),
    "IDRiD labels":       Path("/content/data/idrid/labels.csv"),
}

for name, path in checks.items():
    status = "OK" if path.exists() else "MISSING"
    print(f"  [{status}] {name}: {path}")
EOF
```

---

## Licence summary

| Dataset | Licence type | Commercial use |
|---|---|---|
| APTOS 2019 | Kaggle competition terms | No |
| EyePACS (Kaggle 2015) | Kaggle competition terms | No |
| Messidor-2 | Non-commercial research (registration) | No |
| IDRiD | CC-style research licence (registration) | No |

All datasets are used under their published licences. No patient images are uploaded to any cloud service.
