# 🔢 MNIST Handwritten Digits Classification — Deep Neural Networks

## ✍️ Author
**Oklaliotis Pantelis-Panagiotis**

---

## 📌 Overview

This repository contains an end-to-end Deep Learning implementation for handwritten digit recognition using **TensorFlow** and **Keras** on the classic **MNIST** dataset.

The dataset consists of **70,000** grayscale images of handwritten digits (0–9) of dimensions $28 \times 28$ pixels:
- 🏋️ **Training Set:** 60,000 images and target labels
- 🧪 **Test Set:** 10,000 images and target labels

**Goal:** Design, tune, and evaluate Deep Feedforward Neural Networks (Multilayer Perceptrons) to achieve high classification accuracy while addressing common deep learning challenges such as vanishing gradients, training latency, and overfitting.

---

## 🔄 Workflow & Data Pipeline

The experimental workflow follows a systematic machine learning lifecycle:

1. **Environment Reproducibility:** Fixed random seeds across Python's `random`, `numpy`, and `tensorflow` to ensure fully deterministic and reproducible experiments.
2. **Data Exploration:** Visualized random sample batches of digits with their ground truth class labels via Matplotlib.
3. **Data Preprocessing & Normalization:**
   - **Feature Reshaping:** Flattened each $28 \times 28$ 2D matrix into a 1D vector of **784** features suitable for dense layers.
   - **Intensity Scaling:** Normalized raw pixel intensities from `[0, 255]` to the `[0.0, 1.0]` range to accelerate gradient convergence and stabilize backpropagation.
   - **Label Encoding:** Converted integer target labels into 10-dimensional **One-Hot Encoded** vectors (`tf.one_hot(..., depth=10)`).
4. **Model Architecture Design:** Developed baseline, tuned, and regularized deep neural network models.
5. **Hyperparameter Tuning:** Systematically evaluated activation functions, network depths, learning rates, and optimizers.
6. **Error Analysis & Validation:** Identified and visualized misclassified test samples across each individual digit class to diagnose decision boundaries.

---

## 🏗️ Neural Network Architecture

The final optimized model is a deep multilayer perceptron regularized with Dropout to avoid overfitting while maximizing generalization accuracy on unseen test data.

### Final Model Layer Breakdown

| Layer | Type | Output Shape | Param # | Details & Activation |
|---|---|---|---|---|
| `input` | `InputLayer` | `(None, 784)` | 0 | Flattened $28 \times 28$ input vector |
| `hidden-1` | `Dense` | `(None, 256)` | 200,960 | Fully-connected layer, **ReLU** |
| `dropout` | `Dropout` | `(None, 256)` | 0 | Dropout rate = 0.2 (20% neurons dropped) |
| `hidden-2` | `Dense` | `(None, 128)` | 32,896 | Fully-connected layer, **ReLU** |
| `dropout_1` | `Dropout` | `(None, 128)` | 0 | Dropout rate = 0.2 (20% neurons dropped) |
| `hidden-3` | `Dense` | `(None, 64)` | 8,256 | Fully-connected layer, **ReLU** |
| `dropout_2` | `Dropout` | `(None, 64)` | 0 | Dropout rate = 0.2 (20% neurons dropped) |
| `outputs` | `Dense` | `(None, 10)` | 650 | Output layer, **Softmax** probability distribution |

- **Total Parameters:** 242,762 (948.29 KB)
- **Trainable Parameters:** 242,762
- **Non-trainable Parameters:** 0

---

## ⚙️ Hyperparameters & Optimization Strategy

Three model iterations were analyzed to demonstrate the impact of structural choices and workflow optimizations:

| Parameter | 1. Baseline Model | 2. Tuned Architecture | 3. Final Optimized Model |
|---|---|---|---|
| **Input Scaling** | Unnormalized `[0, 255]` | Unnormalized `[0, 255]` | Normalized `[0.0, 1.0]` |
| **Hidden Layers** | 2 layers: 256 → 256 | 3 layers: 256 → 128 → 64 | 3 layers: 256 → 128 → 64 |
| **Activation Function** | `tanh` | `relu` | `relu` |
| **Regularization** | None | None | **Dropout (0.2)** after each hidden layer |
| **Optimizer** | Plain SGD ($\eta = 0.001$) | Plain SGD ($\eta = 0.001$) | **SGD with Momentum** ($\eta = 0.001, \beta = 0.9$) |
| **Loss Function** | Categorical Cross-Entropy | Categorical Cross-Entropy | Categorical Cross-Entropy |
| **Training Epochs** | 10 | 20 | 30 |

### Key Improvements
- **ReLU vs. Tanh:** Mitigated saturation in deeper layers, enabling faster learning and higher representational capacity.
- **SGD with Momentum (0.9):** Dampened oscillations during gradient descent and accelerated traversal across flat loss plateaus.
- **Feature Normalization:** Stabilized weight updates, significantly reducing the loss value and training variance.
- **Dropout (0.2):** Prevented co-adaptation of hidden units, effectively eliminating the severe overfitting observed in the unregularized 20-epoch model.

---

## 📊 Experimental Results

| Model Configuration | Epochs | Training Acc. | Validation Acc. | Validation Loss | Observations |
|---|:---:|:---:|:---:|:---:|---|
| **1. Baseline Model** | 10 | 93.99% | 93.58% | 0.2201 | Slow convergence; constrained by `tanh` saturation and unnormalized pixel scales. |
| **2. Tuned Architecture** | 20 | 99.50% | 95.73% | 0.2018 | Overfitting: high training accuracy with stagnant test performance and diverging loss. |
| **3. Final Optimized Model** | 30 | **98.29%** | **98.02%** | **0.0679** | **Best performance:** generalises robustly with an ultra-low validation loss and no overfitting gap. |

---

## 🔍 Validation & Error Analysis

To understand the failure modes of the trained classifier:
1. Ground truth test labels were compared against model predictions generated via `argmax(model.predict(x_test))`.
2. One representative misclassified digit was isolated for each digit class ($0$ through $9$).
3. The misclassified examples were rendered in a $2 \times 5$ subplot grid displaying both the true label and the model's prediction.

Most prediction errors stemmed from extreme handwriting ambiguity, abnormal stroke angles, or incomplete digit contours (such as confusing $4$ with $9$, or $3$ with $5$).

---

## 📂 Files

| File | Description |
|---|---|
| `mnist_classification_dnn.ipynb` | Complete Google Colab notebook with exploratory analysis, baseline model, hyperparameter experiments, and final optimized network |
| `report.pdf` | Comprehensive PDF report detailing theoretical background, methodologies, plots, and experimental conclusions |
| `project_description.pdf` | Academic course assignment prompt and project requirements |
| `requirements.txt` | Package and runtime dependencies (TensorFlow, NumPy, Pandas, Matplotlib) |
| `README.md` | Project overview, technical documentation, and model evaluation summary |

---

## 📄 Documentation

You can view the full documentation as a PDF [here](./report.pdf).

---

## 🚀 Environment & Dependencies

To reproduce the experiments locally or on Google Colab, ensure Python 3.10+ is installed along with the required libraries:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `tensorflow >= 2.15.0`
- `numpy`
- `matplotlib`
- `pandas`
