# TrafficVision AI

**Automated Traffic Violation Detection System using Computer Vision**

Built for Flipkart Gridlock Hackathon 2.0 - Round 2 — Theme: Automated Photo Identification and Classification for Traffic Violations Using Computer Vision

## What it does

TrafficVision AI processes traffic images and automatically detects helmet violations, triple riding, wrong-side driving, and illegal parking using a fine-tuned YOLOv8 pipeline. It reads license plates via OCR, generates digital e-challan PDFs with QR codes, blurs faces for privacy compliance, tracks repeat offenders, and provides zone-wise enforcement analytics — closing the loop from raw camera footage to an actionable enforcement document.



## Demo Video
https://youtu.be/GBwG_eq4SUA 


## Architecture

```
Image/Video Input
      ↓
Preprocessing (CLAHE, denoise, night/rain mode)
      ↓
YOLOv8 Detection (vehicles, people) + Pose Estimation + Fine-tuned Helmet Model
      ↓
Violation Logic (helmet, triple riding, wrong-side, illegal parking via DeepSORT tracking)
      ↓
License Plate OCR (EasyOCR) + Face Blurring (privacy)
      ↓
SQLite Logging + Repeat Offender Detection
      ↓
E-Challan PDF Generation (QR code, evidence image, AI reasoning)
      ↓
Streamlit Dashboard (Live Demo, Citizen Reporting, Evidence Log, Analytics, Metrics)
```

## Key Features

- **Real helmet detection model** — fine-tuned YOLOv8s on a public helmet dataset (78.2% mAP50), not a heuristic
- **Explainable AI** — every violation includes a human-readable reason, not just a confidence score
- **Privacy by design** — automatic face blurring on all evidence images
- **Multi-violation per frame** — detects multiple violation types in a single image
- **Zone-based risk ranking** — prioritizes enforcement by severity-weighted zone analytics, not raw counts
- **Repeat offender tracking** — flags plates with 3+ prior violations
- **Citizen reporting mode** — crowdsourced violation submission using the same detection pipeline
- **Auto-generated e-challans** — QR-coded PDF documents, ready for verification

## Tech Stack

| Component | Technology |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Pose estimation | YOLOv8-pose |
| Helmet classification | Custom fine-tuned YOLOv8s |
| Vehicle tracking | DeepSORT |
| License plate OCR | EasyOCR |
| Dashboard | Streamlit |
| Charts | Plotly |
| PDF generation | ReportLab |
| Database | SQLite |

## Model Performance

Fine-tuned helmet detection model, evaluated on a held-out validation set:

| Metric | With Helmet | Without Helmet | Overall |
|---|---|---|---|
| Precision | 86.6% | 77.0% | 81.8% |
| Recall | 82.6% | 64.0% | 73.3% |
| mAP50 | 87.2% | 69.3% | 78.2% |

Inference speed: ~5.6ms/image on NVIDIA RTX 3050.

## Setup & Run Locally

### Prerequisites
- Python 3.10+
- (Optional but recommended) NVIDIA GPU with CUDA for faster inference

### Installation

```bash
git clone https://github.com/kumar1035/traffic_violence.git
cd traffic_violence

python -m venv traffic_env
traffic_env\Scripts\activate          # Windows


pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`. First run auto-downloads base YOLOv8 weights (~100MB total).

## Known Limitations (Honest Disclosure)

- Helmet detection accuracy decreases significantly on aerial/distant camera angles where head regions are too small to classify reliably — this is a genuine model generalization constraint, not a logic bug
- License plate OCR works best on close, front/rear-facing, well-lit vehicle shots; accuracy drops on angled, distant, or motion-blurred plates
- Wrong-side driving and illegal parking detection require video input (multi-frame tracking) and are not available for single-image uploads
- Trained on a relatively small dataset (126 validation images) due to hackathon time constraints — production deployment would benefit from a larger, India-specific training set (e.g. IDD)



## Team Name
-minecraft
[Anuj Kumar]
[Sujal Bhatu]
