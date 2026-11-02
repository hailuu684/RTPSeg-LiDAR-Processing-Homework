# RTPSegNet: LiDAR Semantic Segmentation (Reproduction & DINOv2 Adaptation)

This repository contains the reproduction of the **RTPSegNet** framework alongside architectural modifications tailored for LiDAR and multi-modal semantic segmentation. In addition to verifying the author's original setup, this project investigates eliminating the Thermal Infrared (TIR) input stream and integrating **DINOv2** as the primary vision encoder backbone.

Comprehensive experiment tracking, hyperparameter setups, and loss/metric curves are documented in our interactive Weights & Biases report:
- 📊 **[Weights & Biases Interactive Experiment Report](https://forge.coreweave.com/wandb/deep-learning-lab-msc/rtpsegnet/reports/Lidar-Processing-RTPSegNet-Report--VmlldzoxODE3MzM2NQ?accessToken=4dlb8d5jlaejgs81ulwj071kihhy17zhj2pvxcobdnjyjzq0xr37z6t87j67q0mc)**

---

## 📌 Key Contributions & Architectural Changes

1. **Baseline Reproduction**: Successfully reproduced the original RTPSegNet benchmarks on point cloud / multi-modal benchmarks.
2. **Elimination of TIR Channel**:
   - Stripped thermal infrared (TIR) dependency from the input pipeline and fusion layers.
   - Reduced deployment sensor constraints to RGB-only and LiDAR modalities without sacrificing core segmentation capabilities.
3. **DINOv2 Encoder Integration**:
   - Replaced the baseline feature extraction backbone with foundation vision transformer features from **DINOv2**.
   - Leveraged self-supervised pre-trained representations for richer semantic cue extraction and dense feature alignment.

---

## 📂 Project Structure

```text
RTPSegNet/
├── checkpoints/       # Trained model weights (.pth) - excluded from git
├── config/            # YAML/JSON training & model configuration files
├── dataloader/        # Dataset parsing, augmentation, and point projection
├── model/             # Modified network modules, fusion blocks & DINOv2 wrappers
├── network/           # Core architecture definition
├── results/           # Evaluation logs and predictions - excluded from git
├── scripts/           # Automation pipeline scripts
│   ├── rtpseg/        # Data download, preprocessing, training, and evaluation
│   └── toronto3d/     # Dataset-specific scripts
├── tools/             # Helper utilities and data conversion scripts
├── utils/             # Metric computation, loss functions, and logging helpers
├── evaluate.py        # Model inference and evaluation entrypoint
├── train.py           # Training routine
└── requirements.txt   # Python dependencies
```

## ⚙️ Environment Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/hailuu684/RTPSeg-LiDAR-Processing-Homework.git](https://github.com/hailuu684/RTPSeg-LiDAR-Processing-Homework.git)
   cd RTPSeg-LiDAR-Processing-Homework
   ```

2. **Create and activate a virtual environment:**
   ```bash
   conda create -n rtpseg python=3.10 -y
   conda activate rtpseg
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Training & Evaluation

Automated shell scripts for the full workflow are provided under `scripts/rtpseg/`:

### 1. Download & Preprocess Data
```bash
bash scripts/rtpseg/01_download.sh
bash scripts/rtpseg/02_unzip.sh
bash scripts/rtpseg/03_preprocess.sh
```

*(Optional) Validate multi-modal projection and point visualization:*
```bash
bash scripts/rtpseg/04_visualize.sh
```

### 2. Train the Model
To train the baseline or the modified DINOv2 (RGB-only) architecture:
```bash
# Using the automated shell script
bash scripts/rtpseg/05_train.sh

# Or directly with Python
python train.py --config config/your_config.yaml
```

### 3. Evaluate the Model
To run inference and compute segmentation metrics (mIoU, Precision, Recall):
```bash
# Run validation evaluation
bash scripts/rtpseg/06_evaluate.sh

# Run author baseline checkpoint comparison
bash scripts/rtpseg/07_evaluate_author.sh
```
