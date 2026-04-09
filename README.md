# 🔍 AI vs Real Image Detection — DWDM Project

> **Data-Driven Detection and Analysis of AI-Generated vs Real Visual Media**  
> A complete end-to-end machine learning pipeline built for the Data Warehousing & Data Mining (DWDM) course — 2024–25

---

## 📌 Project Overview

With the rapid rise of AI image generators like Midjourney, Stable Diffusion, and DALL-E,
detecting synthetic images has become a critical challenge in digital forensics,
misinformation detection, and content moderation.

This project builds a **binary image classifier** that distinguishes between:
- 🤖 **AI-Generated Images** — produced by GANs, diffusion models, and other generative AI
- 📷 **Real Photographs** — sourced from COCO, Flickr30k, ImageNet, and Open Images

We go beyond just training a model — the project includes a full **DWDM pipeline**:
data collection → cleaning → EDA → GAN generation → model training → evaluation → inference.

---

## 🏆 Results

| Model | Accuracy | AUC-ROC | Avg Precision |
|-------|----------|---------|---------------|
| **ResNet50** (Transfer Learning) | **96.93%** | **0.9939** | **0.9937** |
| Random Forest (15 features) | 75.55% | 0.8283 | 0.7754 |
| **Ensemble** (soft voting) | **~97%** | **~0.995** | — |

> Best checkpoint saved at Epoch 17 (Val AUC = 0.9946)

---

## 📁 Project Structure
DWDM-AI-vs-Real-Image-Detection/
│
├── 📂 dataset/
│   ├── ai_images/                  ← Raw AI-generated images
│   └── real_images/                ← Raw real photographs
│
├── 📂 cleaned_data/
│   ├── standardised/               ← 224×224 RGB JPEG images
│   ├── splits/                     ← Train / Val / Test splits
│   │   ├── train/  (ai/ + real/)
│   │   ├── val/    (ai/ + real/)
│   │   └── test/   (ai/ + real/)
│   └── eda_plots/                  ← EDA visualisations
│
├── 📂 models/
│   ├── resnet50_best.pth           ← Best ResNet50 checkpoint
│   ├── random_forest.pkl           ← Trained Random Forest
│   ├── gan_generator_final.pth     ← Trained DCGAN Generator
│   └── training_curves.png
│
├── 📂 gan_outputs/
│   ├── training_samples/           ← GAN sample grids per epoch
│   └── generated_images/           ← Final GAN-generated images
│
├── step1_data_audit.py             ← Scan & audit raw dataset
├── step2_validate_images.py        ← Detect corrupt/invalid images
├── step3_remove_duplicates.py      ← MD5 + pHash deduplication
├── step4_standardise_images.py     ← Resize, convert, strip EXIF
├── step5_eda.py                    ← Exploratory data analysis
├── step6_train_val_test_split.py   ← Stratified 70/15/15 split
│
├── gan_generator.py                ← DCGAN training & generation
├── gan_evaluate.py                 ← GAN quality evaluation + integration
├── train_model.py                  ← Train ResNet50 + Random Forest
├── evaluate_model.py               ← Test set evaluation + plots
├── predict.py                      ← CLI inference on new images
├── quick_check.py                  ← Interactive image checker
│
├── requirements.txt
└── README.md
---

## 🗂️ Dataset

| Class | Count | Sources |
|-------|-------|---------|
| AI-Generated | 11,845 | Midjourney v5/v6, Stable Diffusion 1.5 & 2.1, DALL-E 3, StyleGAN2/3, Kaggle datasets |
| Real Photos | 9,433 | COCO 2017, Flickr30k, ImageNet, Open Images v7 |
| **Total** | **21,278** | — |

**Split:** Train = 14,894 · Val = 3,190 · Test = 3,194 (stratified 70/15/15)

---

## 🧹 Data Cleaning Pipeline

A 6-step reproducible pipeline — every decision logged to CSV/JSON:

| Step | Script | What It Does |
|------|--------|-------------|
| 1 | `step1_data_audit.py` | Scan folders, count files, check formats & sizes |
| 2 | `step2_validate_images.py` | Detect corrupt, truncated, palette-mode images |
| 3 | `step3_remove_duplicates.py` | MD5 exact + pHash near-duplicate removal |
| 4 | `step4_standardise_images.py` | Resize 224×224, RGB JPEG, strip EXIF metadata |
| 5 | `step5_eda.py` | Pixel stats, brightness, RGB & size distributions |
| 6 | `step6_train_val_test_split.py` | Stratified split + manifest CSV |

**Cleaning results:** 0 corrupt files · 0 duplicates · 192 warnings handled · all images → 224×224 RGB

---

## 🤖 GAN — DCGAN Generator

A **Deep Convolutional GAN** trained on AI images to generate new synthetic samples for data augmentation.
Noise(100) → 4×4 → 8×8 → 16×16 → 32×32 → 64×64 RGB image
- Generator: 3.57M parameters · 4 transposed conv blocks · BatchNorm + ReLU + Tanh
- Discriminator: 2.76M parameters · 4 conv blocks · LeakyReLU + Sigmoid
- Training: 100 epochs · AdamW · label smoothing · LR = 2e-4

```bash
python gan_generator.py --mode train
python gan_generator.py --mode generate --num_images 500
```

---

## 🧠 Models

### ResNet50 (Transfer Learning)
- Pretrained on ImageNet-1K
- Custom binary classifier head (FC-256 + Dropout)
- Two-phase training: frozen backbone → full fine-tuning
- 20 epochs · AdamW · Cosine Annealing · Label smoothing

### Random Forest
- 15 handcrafted pixel features per image
- Features: RGB means/stds, brightness, entropy, Laplacian variance, IQR
- 300 trees · balanced class weights

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# GPU version of PyTorch (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Run Cleaning Pipeline
```bash
python step1_data_audit.py
python step2_validate_images.py
python step3_remove_duplicates.py
python step4_standardise_images.py
python step5_eda.py
python step6_train_val_test_split.py
```

### 3. Train GAN (optional — for data augmentation)
```bash
python gan_generator.py --mode train
python gan_generator.py --mode generate --num_images 500
python gan_evaluate.py --integrate
```

### 4. Train Classifier
```bash
python train_model.py
```

### 5. Evaluate
```bash
python evaluate_model.py
```

### 6. Check a New Image
```bash
# Interactive (easiest)
python quick_check.py

# Command line
python predict.py --image "path/to/image.jpg"
python predict.py --image "path/to/image.jpg" --model ensemble
python predict.py --folder "path/to/folder/" --output results.csv
```

---

## 📊 Key Visualisations

| Plot | Description |
|------|-------------|
| `training_curves.png` | ResNet50 loss & accuracy over 20 epochs |
| `eval_resnet50_roc.png` | ROC curve — AUC = 0.9939 |
| `eval_rf_roc.png` | ROC curve — AUC = 0.8283 |
| `model_comparison.png` | Side-by-side accuracy / AUC / precision |
| `eval_rf_confidence.png` | Confidence distribution: correct vs wrong |
| `eda_plots/` | Brightness, RGB channel, file size distributions |

---

## 👥 Team

| Name | Role |
|------|------|
| **Rahul** | Data Collection & Preprocessing |
| **Mandeep** | Data Cleaning & EDA |
| **Shashi Kant** | Model Training & Evaluation |
| **Akshat** | Feature Engineering & Random Forest |
| **Prathamesh** | ResNet50 CNN & Report Writing |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia)

- **Deep Learning:** PyTorch, torchvision, ResNet50, DCGAN
- **Traditional ML:** scikit-learn, Random Forest
- **Image Processing:** Pillow, NumPy, imagehash
- **Visualisation:** Matplotlib, Seaborn
- **Data:** COCO, Flickr30k, ImageNet, Open Images, Kaggle

---

## 📄 License

This project is submitted as an academic project for the DWDM course (2024–25).  
For educational use only.

---

> *"In an era where seeing is no longer believing, data mining becomes the new truth detector."*
