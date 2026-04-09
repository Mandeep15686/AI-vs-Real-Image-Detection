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
