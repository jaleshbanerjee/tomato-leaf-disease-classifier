# 🍅 Tomato Leaf Disease Classifier

> A deep learning project to detect tomato leaf diseases using **MobileNetV2 + Transfer Learning**, with GradCAM explainability visualizations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mf1hw-CmIm-Ppbyb53wzSwjyr-OwdweE?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

### 🚀 Transfer Learning · GradCAM Explainability · Data Augmentation · Model Evaluation

---

## ✨ Features

- 🧠 **MobileNetV2** pretrained on ImageNet, fine-tuned on PlantVillage tomato dataset
- 🔥 **GradCAM explainability** — visualizes exactly which leaf regions the model focuses on
- 📊 **Full evaluation suite** — accuracy, precision, recall, confusion matrix, per-class report
- 🎯 **92.71% test accuracy** across 3 disease classes
- 🔄 **Data augmentation** — rotation, flip, zoom, shift for robust generalization
- 💾 **Pretrained weights** available via Google Drive for instant inference
- 📈 **Training curves** with early stopping and learning rate scheduling

---

## 📌 Overview

This project classifies tomato leaf images into **3 categories** using a fine-tuned MobileNetV2 model trained on the PlantVillage dataset.

| Class | Label |
|-------|-------|
| ✅ Healthy | `Tomato___healthy` |
| 🟡 Early Blight | `Tomato___Early_blight` |
| 🔴 Late Blight | `Tomato___Late_blight` |

---

## 🧠 Model Architecture

```text
Input (150×150×3)
└── MobileNetV2 (pretrained on ImageNet, frozen)
└── GlobalAveragePooling2D
└── Dense(64, ReLU)
└── Dropout(0.5)
└── Dense(3, Softmax)
```

- **Base model:** MobileNetV2 (weights = ImageNet)
- **Fine-tuning:** Top layers trainable, base frozen
- **Optimizer:** Adam (lr=1e-3) with ReduceLROnPlateau
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **92.71%** |
| Test Precision | **93.09%** |
| Test Recall | **92.60%** |

### Training Curves
![Training Curves](results/training_curves.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

---

## 🔍 GradCAM Explainability

GradCAM (Gradient-weighted Class Activation Mapping) visualizes **which regions of the leaf the model focuses on** when making predictions.

![GradCAM Heatmaps](results/gradcam_samples.png)

> The heatmap highlights diseased regions in red/yellow, showing the model has genuinely learned pathological patterns — not just background noise.

---

## 🗂️ Dataset

- **Source:** [PlantVillage Tomato Dataset — Kaggle](https://www.kaggle.com/datasets/charuchaudhry/plantvillage-tomato-leaf-dataset)
- **Total images:** ~3200
- **Split:** 70% train / 15% val / 15% test

![Class Distribution](results/class_distribution.png)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-orange?style=flat&logo=tensorflow&logoColor=white) | Model training & inference |
| ![Keras](https://img.shields.io/badge/Keras-red?style=flat&logo=keras&logoColor=white) | High-level model building API |
| ![OpenCV](https://img.shields.io/badge/OpenCV-green?style=flat&logo=opencv&logoColor=white) | GradCAM heatmap overlay |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Metrics & evaluation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Array operations |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat) | Visualization & plots |
| ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white) | Dataset source |

---

## 📁 Project Structure

```text
tomato-leaf-disease-classifier/
├── notebooks/
│   └── tomato_leaf_disease_classifier.ipynb
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── gradcam_samples.png
│   └── class_distribution.png
├── reports/
│   └── project_report.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 How to Run

1. Open the notebook in Google Colab using the badge above
2. Upload your `kaggle.json` API key when prompted
3. Run all cells top to bottom

**Install dependencies locally:**
```bash
pip install -r requirements.txt
```

---

## 💾 Pretrained Model

The trained model weights (`best_model.h5`) are available here:
👉 [Download from Google Drive](https://drive.google.com/file/d/1ZU3KfAm1yN0Ka_ryg9elqiGHvlca-s8x/view?usp=sharing)

---

## 📜 License

This project is licensed under the MIT License.
