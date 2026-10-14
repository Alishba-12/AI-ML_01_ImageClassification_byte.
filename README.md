# 🐱🐶 Cats vs Dogs Image Classifier

> A deep learning image classifier that distinguishes between cats and dogs using transfer learning with ResNet18, trained on the full 25,000-image Microsoft dataset and achieving 98.18% test accuracy.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

---

## 📊 Model Performance

### Best Model: ResNet18 (Transfer Learning)

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **98.18%** |
| **Validation Loss** | 0.0532 |
| **Precision** | 0.9829 |
| **Recall** | 0.9810 |
| **F1-Score** | 0.9819 |

### Confusion Matrix

| | Predicted Cat | Predicted Dog |
|---|---|---|
| **Actual Cat** | 2435 | 43 |
| **Actual Dog** | 48 | 2474 |

- **True Negatives (Cat correct):** 2435
- **False Positives (Cat predicted as Dog):** 43
- **False Negatives (Dog predicted as Cat):** 48
- **True Positives (Dog correct):** 2474

Out of 5,000 validation images, only 91 were misclassified.

---

## 🎯 Features

- Transfer learning with pre-trained ResNet18 on ImageNet
- Full 24,998-image Microsoft Cats vs Dogs dataset
- Data augmentation (random flip, rotation, color jitter)
- Learning rate scheduling with ReduceLROnPlateau
- Comprehensive evaluation: accuracy, precision, recall, F1, confusion matrix
- Google Colab compatible (no local setup required)

---


## 🚀 Quick Start

### Option 1: Run in Google Colab

Open the notebook in Colab, enable GPU (Runtime → Change runtime type → T4 GPU), then run all cells.

### Option 2: Run Locally

git clone https://github.com/Alishba-12/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
pip install -r requirements.txt
jupyter notebook Cats_vs_Dogs.ipynb

### 🛠️ Technologies Used
Technology	Purpose
PyTorch	Deep learning framework
torchvision	Pre-trained models and transforms
ResNet18	Backbone architecture
Matplotlib / Seaborn	Visualization
scikit-learn	Evaluation metrics
Google Colab (T4 GPU)	Training environment
### 📄 Dataset
Source: Microsoft Cats vs Dogs Dataset (Kaggle)

### Statistics:

Metric	Value
Total images	24,998
Train set	19,998 (80%)
Validation set	5,000 (20%)
Classes	Cat, Dog
Image size	Resized to 224×224

### 📈 Training Configuration
Parameter	Value
Model	ResNet18 (ImageNet pre-trained)
Epochs	15
Batch size	32
Optimizer	Adam
Learning rate	0.001
Scheduler	ReduceLROnPlateau (patience=3)
Loss function	CrossEntropyLoss
Augmentation	Random flip, rotation ±10°, color jitter

### 🔍 How It Works
Pipeline
text
Image → Resize (224×224) → Normalize → Augment → ResNet18 → Softmax → Cat / Dog
Model Architecture
Backbone: ResNet18 with ImageNet weights

Final layer: Replaced with Linear(512, 2) for binary classification

Training: Fine-tuned all layers on the Cats vs Dogs dataset

Training Progress
Epoch 1: Val Accuracy ~94%

Epoch 6: Val Accuracy ~97.6%

Epoch 15: Val Accuracy ~98.2%

Both training and validation loss decreased steadily, showing no significant overfitting.

⚠️ Limitations
Occasional misclassification on images with both animals or obscured faces (91/5000 in validation)

Trained only on cats and dogs; does not generalize to other animals

Sensitive to extreme lighting or unusual poses

🚀 Future Improvements
Try ResNet34 or EfficientNet for even higher accuracy

Deploy as a web app (Streamlit or Gradio)

Add explainability with Grad-CAM visualizations

Build an inference API with FastAPI

👩‍💻 Author
Alishba

https://img.shields.io/badge/GitHub-Alishba--12-blue?style=flat&logo=github

⭐ If you find this useful, please give it a star.

