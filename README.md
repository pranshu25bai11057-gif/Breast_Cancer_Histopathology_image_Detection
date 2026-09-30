# 🩺 Project Exhibition 01

# Breast Cancer Detection Using Histopathology Images and Machine Learning

An AI/ML-based application for classifying breast histopathology images as **Benign**, **Malignant**, or **Invalid** using a **Convolutional Neural Network (CNN)**. The application includes image validation, unified preprocessing, a saved Keras model, and a Streamlit interface.

---

## 📑 Table of Contents

* [Project Overview](#-project-overview)
* [Problem Statement](#-problem-statement)
* [Objectives](#-objectives)
* [How the System Works](#-how-the-system-works)
* [System Workflow](#-system-workflow)
* [Technology Stack](#-technology-stack)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Running the Application](#-running-the-application)
* [Model Prediction](#-model-prediction)
* [CNN Architecture](#-cnn-architecture)
* [Features](#-features)
* [Model Evaluation](#-model-evaluation)
* [Testing](#-testing)
* [Team & Contributions](#-team--contributions)
* [Limitations](#-limitations)
* [Future Scope](#-future-scope)
* [Conclusion](#-conclusion)

---

# 🔬 Project Overview

Breast cancer histopathology images contain visual cellular and tissue patterns that can be difficult to interpret consistently. This project demonstrates the use of **Machine Learning and Deep Learning** for automated classification of breast histopathology images.

The current system accepts a histopathology image, verifies that it can be read, preprocesses it, and passes it through a trained **4-stage Convolutional Neural Network (CNN)**.

The model produces one of three classifications:

* 🟢 **Benign**
* 🔴 **Malignant**
* 🟠 **Invalid** - non-histopathology or unsuitable input class

Along with the predicted class, the application displays the model's **output confidence** and the full probability distribution across the three classes.

> **Note:** This project is developed for educational and research purposes. It is not intended to replace professional medical diagnosis.

---

# Problem Statement

Histopathology analysis involves examining tissue patterns under microscopy and can require substantial expert effort. This project demonstrates how a compact deep-learning image-classification pipeline can provide an automated preliminary classification of breast histopathology inputs while also handling unsuitable inputs.

The project is intentionally an educational/research prototype and is **not** designed for clinical diagnosis or treatment decisions.

---

# ✅ Objectives

The main objectives of this project are:

1. Develop a reproducible CNN-based histopathology image-classification model.
2. Classify breast histopathology images into **Benign**, **Malignant**, and **Invalid** categories.
3. Validate image files before model inference.
4. Apply the same core preprocessing path during training and inference.
5. Provide prediction results with model confidence and class probabilities.
6. Develop a simple and user-friendly web interface using Streamlit.
7. Evaluate performance on a patient-level separated test set.
8. Demonstrate an end-to-end AI/ML workflow including data preparation, training, evaluation, inference, and testing.

---

# How the System Works

The current application follows these stages:

### 1. Image Input

The user can either:

* Upload a PNG, JPG, or JPEG image, or
* Select one of the built-in demonstration samples.

### 2. Image Validation

The system attempts to open and verify the image. Corrupted, unsupported, or unreadable files are handled before CNN inference.

The trained model also contains an **Invalid** output class for non-histopathology images.

### 3. Image Preprocessing

The image is:

* Converted to RGB
* Resized to **128 × 128** pixels
* Converted into a numerical array
* Normalized to the **[0, 1]** range
* Expanded to a batch shape of **(1, 128, 128, 3)**

### 4. CNN Prediction

The preprocessed image is passed to the trained 4-stage CNN stored as:

```text
model/breast_cancer_model.keras
```

### 5. Result Display

The Streamlit application displays:

**Predicted Class**

→ Benign / Malignant / Invalid

**Confidence**

→ Top model output for the predicted class.

**Probability Distribution**

→ Probability values for all three classes.

---

# System Workflow

```text
                    ┌────────────────────┐
                    │  User Upload /     │
                    │  Demo Sample       │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Image Verification │
                    └─────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
               Invalid/File OK      Corrupt/Unsupported
                    │                   │
                    ▼                   ▼
             ┌──────────────┐      ┌──────────────┐
             │ Preprocessing│      │ Error/Invalid│
             └──────┬───────┘      └──────────────┘
                    │
                    ▼
             ┌──────────────┐
             │ 4-Stage CNN  │
             └──────┬───────┘
                    │
                    ▼
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
    Benign       Invalid     Malignant
       │            │            │
       └────────────┼────────────┘
                    ▼
           Confidence +
        Probability Distribution
```

---

# Technology Stack

| Technology | Purpose |
| --- | --- |
| **Python** | Main programming language |
| **TensorFlow / Keras** | CNN development, training, and model inference |
| **NumPy** | Numerical and image-array operations |
| **OpenCV** | Image-processing dependency |
| **Pillow (PIL)** | Image loading, validation, conversion, and resizing |
| **Scikit-learn** | Classification metrics and class-weight calculation |
| **Matplotlib** | Evaluation visualizations |
| **Streamlit** | Web application interface |

---

# Project Structure

The current repository keeps the runnable application and trained model available for inference, while the data-preparation and evaluation scripts are also included for reproducibility.

```text
Breast_Cancer_Histopathology_image_Detection/
│
├── app.py
├── predict.py
├── requirements.txt
├── .gitattributes
├── Project Report.pdf
├── README.md
├── MODEL_EVALUATION_REPORT.md
│
├── model/
│   └── breast_cancer_model.keras
│
├── sample_images/
│   ├── sample_benign.png
│   ├── sample_benign_adenoma.png
│   ├── sample_benign_fibroadenoma.png
│   ├── sample_invalid.png
│   ├── sample_malignant.png
│   ├── sample_malignant_ductal.png
│   └── sample_malignant_mucinous.png
│
└── breast-cancer-ml/
    ├── train_model.py
    ├── prepare_data.py
    ├── evaluate_detailed.py
    ├── test_system.py
    └── requirements.txt
```

> The raw BreaKHis dataset is not required to run the saved-model application. The training scripts expect the prepared dataset structure described in `prepare_data.py`.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/pranshu25bai11057-gif/Breast_Cancer_Histopathology_image_Detection.git
```

Navigate to the repository:

```bash
cd Breast_Cancer_Histopathology_image_Detection
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
py -3 -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The application dependencies include TensorFlow, Pillow, NumPy, scikit-learn, Matplotlib, OpenCV, and Streamlit.

---

# Running the Application

From the repository root, run:

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, normally:

```text
http://localhost:8501
```

Open that URL in your browser if it does not launch automatically.

---

# Model Prediction

The current prediction pipeline is:

```text
Input Image
     ↓
Image Verification
     ↓
RGB Conversion
     ↓
Resize to 128 × 128
     ↓
Normalize to [0, 1]
     ↓
4-Stage CNN
     ↓
3-Class Softmax Output
     ↓
Benign / Invalid / Malignant
     ↓
Confidence + Probability Distribution
```

The saved Keras model is loaded from:

```text
model/breast_cancer_model.keras
```

The inference code maps the three output neurons to:

```text
['benign', 'invalid', 'malignant']
```

Corrupted or unreadable files are trapped before CNN inference and returned as an invalid/error result.

---

# CNN Architecture

The current training code defines a **4-stage CNN** with Batch Normalization, Dropout, Max Pooling, Global Average Pooling, L2 regularization, and a compact classification head.

```text
Input: 128 × 128 × 3
        ↓
Conv2D: 32 filters
BatchNorm → ReLU → MaxPool → Dropout(0.10)
        ↓
Conv2D: 64 filters
BatchNorm → ReLU → MaxPool → Dropout(0.20)
        ↓
Conv2D: 128 filters
BatchNorm → ReLU → MaxPool → Dropout(0.20)
        ↓
Conv2D: 128 filters
BatchNorm → ReLU → MaxPool → Dropout(0.25)
        ↓
Global Average Pooling
        ↓
Dense(64, ReLU, L2 regularization)
        ↓
Dropout(0.30)
        ↓
Dense(3, Softmax)
```

### Training configuration

| Setting | Current value |
| --- | --- |
| Input size | 128 × 128 × 3 |
| Batch size | 64 |
| Maximum epochs | 20 |
| Initial learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Data augmentation | Horizontal/vertical flip, rotation, zoom |
| Class balancing | Balanced class weights |
| Checkpointing | Best validation loss |
| Early stopping | Patience 5 |
| Learning-rate reduction | ReduceLROnPlateau |

> The training script is the source of truth for the current CNN architecture. The application UI may contain older descriptive wording about the number of stages; the current training implementation uses four convolutional stages.

---

# Key Features

### Image Upload

Users can upload PNG, JPG, or JPEG histopathology images directly through the web interface.

### Built-in Demo Samples

The application includes demonstration examples covering benign, malignant, and invalid inputs.

### Input Validation

The prediction utility verifies image integrity and handles corrupted or unsupported files before inference.

### 3-Class CNN Classification

The model predicts **Benign**, **Malignant**, or **Invalid**.

### Confidence Score

The application displays the top model output as a confidence percentage.

### Probability Distribution

An expandable section shows the model probabilities for all three classes.

### Streamlit Interface

The system can be demonstrated through a simple web interface without directly interacting with Python code.

### Patient-Level Evaluation Design

The data-preparation script groups BreaKHis images by patient identifier before creating train, validation, and test splits.

---

# Model Evaluation

The current evaluation uses **1,664 test images** and reports three-class performance across **Benign, Invalid, and Malignant**.

## Overall Performance

| Metric | Score |
| --- | ---: |
| Accuracy | **75.96%** |
| Macro Precision | **80.33%** |
| Weighted Precision | **76.19%** |
| Macro Recall | **78.85%** |
| Weighted Recall | **75.96%** |
| Macro F1-Score | **79.56%** |
| Weighted F1-Score | **76.07%** |

## Per-Class Performance

| Class | Support | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: | ---: |
| Benign | 486 | 58.72% | 60.29% | 59.49% |
| Malignant | 1,103 | 82.27% | 81.60% | 81.93% |
| Invalid | 75 | 100.00% | 94.67% | 97.26% |

## Confusion Matrix

```text
                  Predicted
              Benign  Invalid  Malignant
True Benign     293       0       193
True Invalid      3      71         1
True Malignant  203       0       900
```

## Performance by Magnification

| Magnification | Samples | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40X | 392 | 75.00% | 75.92% | 75.00% | 75.37% |
| 100X | 422 | 77.73% | 78.16% | 77.73% | 77.91% |
| 200X | 389 | 73.01% | 73.07% | 73.01% | 73.04% |
| 400X | 386 | 74.35% | 73.67% | 74.35% | 73.92% |

## Performance by Histopathological Subtype

| Subtype | Type | Samples | Correct | Rate |
| --- | --- | ---: | ---: | ---: |
| Ductal Carcinoma | Malignant | 258 | 258 | 100.00% |
| Fibroadenoma | Benign | 68 | 0 | 0.00% |
| Lobular Carcinoma | Malignant | 125 | 125 | 100.00% |
| Mucinous Carcinoma | Malignant | 361 | 261 | 72.30% |
| Papillary Carcinoma | Malignant | 359 | 256 | 71.31% |
| Phyllodes Tumor | Benign | 235 | 131 | 55.74% |
| Tubular Adenoma | Benign | 183 | 162 | 88.52% |

## Invalid-Input Evaluation

* Invalid test samples: **75**
* Correctly assigned to Invalid: **71 / 75**
* Invalid recall: **94.67%**
* Invalid precision: **100.00%**
* Corrupted/invalid file handling: **Passed in the supplied system tests**

> **Evaluation note:** The repository's `MODEL_EVALUATION_REPORT.md` contains stale hard-coded narrative values that conflict with its computed tables and confusion matrix. The figures above follow the computed metric table and confusion matrix because they are internally consistent.

---

# Testing

The repository includes `test_system.py` for system verification. The test script checks:

1. Model file existence.
2. Successful model loading.
3. Benign test-image inference.
4. Malignant test-image inference.
5. Invalid non-histopathology inference.
6. Corrupted-file handling.
7. Built-in demo sample predictions.
8. Confidence values within the expected 0-100 range.

A successful test run ends with:

```text
ALL SYSTEM VERIFICATION TESTS PASSED
```

---

# Team & Contributions

## Author

### **Shwetank Vaibhav**

---

## Team Members

* **Pranshu Gupta**
* **Ayush Shukla**
* **Debasish Kumar Sahoo**
* **Manvendra Kumar**
* **Piyush Kumar Dash**

The project was developed collaboratively, with team members contributing to areas including:

* Dataset research
* Data preprocessing
* CNN model development
* Model training
* Model evaluation
* Streamlit application development
* Testing
* Documentation
* Project presentation

---

### Deployment

The Streamlit application can be deployed to a suitable cloud platform after local reproducibility and evaluation are stable.

---

# Conclusion

Project Exhibition 01 demonstrates how Artificial Intelligence and Deep Learning can be applied to breast histopathology image classification.

The current system combines:

```text
Image Validation
       +
Image Preprocessing
       +
4-Stage CNN Classification
       +
Confidence / Probability Output
       +
Streamlit Interface
       +
System Verification Tests
```

to create a compact end-to-end machine-learning application capable of classifying inputs as **Benign**, **Malignant**, or **Invalid**.

The current evaluation reports **75.96% three-class accuracy on 1,664 test images**, with **81.60% malignant recall** and **94.67% Invalid-class recall**. The reported subtype results also expose important failure cases, especially within some benign subtypes.

The project is intended as an educational and research demonstration of medical image classification. Its outputs should not be used as a substitute for professional medical evaluation.

---

## Project Highlights

> **AI/ML Project**  
> **Domain:** Medical Image Analysis  
> **Task:** 3-Class Image Classification  
> **Model:** 4-Stage Convolutional Neural Network (CNN)  
> **Classes:** Benign / Malignant / Invalid  
> **Language:** Python  
> **Interface:** Streamlit

---

## Disclaimer

This project is intended **only for educational and research purposes**. Predictions generated by the model should not be used as a substitute for professional medical advice, diagnosis, or treatment.
