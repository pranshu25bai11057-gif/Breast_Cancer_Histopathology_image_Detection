# 🩺 Project Exhibition 01

# Breast Cancer Detection Using Histopathology Images and Machine Learning

An AI/ML-based application for classifying breast histopathology images as **Benign** or **Malignant** using a **Convolutional Neural Network (CNN)**.

---

##  Table of Contents

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
* [Features](#-features)
* [Team & Contributions](#-team--contributions)
* [Limitations](#-limitations)
* [Future Scope](#-future-scope)
* [Conclusion](#-conclusion)

---

##  Project Overview

Breast cancer is one of the most common types of cancer worldwide. Histopathology is an important method used for examining tissue samples and identifying abnormal cellular structures.

This project demonstrates the use of **Machine Learning and Deep Learning** for automated classification of breast histopathology images.

The system accepts a histopathology image from the user, validates the input, preprocesses the image, and passes it through a trained **Convolutional Neural Network (CNN)**.

The model produces one of two classifications:

* 🟢 **Benign**
* 🔴 **Malignant**

Along with the predicted class, the application displays the model's **prediction confidence**.

> **Note:** This project is developed for educational and research purposes. It is not intended to replace professional medical diagnosis.

---

##  Problem Statement

Traditional histopathology analysis requires examination of tissue samples by trained medical professionals. This process can be time-consuming and may involve significant manual effort.

The objective of this project is to demonstrate how **deep learning-based image classification** can assist in analyzing histopathology images and provide an automated preliminary classification.

---

##  Objectives

The main objectives of this project are:

1. Develop a CNN-based image classification model.
2. Classify breast histopathology images into **Benign** and **Malignant** categories.
3. Validate uploaded images before performing prediction.
4. Apply appropriate image preprocessing.
5. Provide prediction results with confidence scores.
6. Develop a simple and user-friendly web interface using Streamlit.
7. Demonstrate an end-to-end AI/ML image-classification workflow.

---

#  How the System Works

The application follows a simple five-stage process:

### 1. Image Upload

The user uploads a breast histopathology image through the Streamlit application.

### 2. Image Validation

The system checks whether the uploaded file is a valid image and suitable for the classification workflow.

Invalid or unsupported inputs are rejected before they reach the model.

### 3. Image Preprocessing

Valid images are prepared according to the requirements of the trained model.

Typical preprocessing includes:

* Image loading
* Image resizing
* Format conversion
* Pixel normalization
* Conversion into a model-compatible array

### 4. CNN Prediction

The preprocessed image is passed to the trained **Convolutional Neural Network**.

The CNN extracts relevant visual features from the image and predicts the corresponding class.

### 5. Result Display

The application displays:

**Predicted Class**

→ Benign / Malignant

**Confidence**

→ Model's confidence in its prediction.

---

#  System Workflow

```text
                    ┌───────────────────┐
                    │   User Uploads    │
                    │      Image        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Image Validation  │
                    └─────────┬─────────┘
                              │
                     ┌────────┴────────┐
                     │                 │
                    NO                YES
                     │                 │
                     ▼                 ▼
             ┌──────────────┐   ┌──────────────┐
             │ Invalid      │   │ Preprocessing│
             │ Image        │   └──────┬───────┘
             └──────────────┘          │
                                       ▼
                               ┌──────────────┐
                               │  CNN Model   │
                               └──────┬───────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ Classification│
                              └──────┬───────┘
                                     │
                           ┌─────────┴─────────┐
                           │                   │
                           ▼                   ▼
                      🟢 Benign          🔴 Malignant
                           │                   │
                           └─────────┬─────────┘
                                     ▼
                              Confidence Score
```

---

#  Technology Stack

| Technology             | Purpose                                   |
| ---------------------- | ----------------------------------------- |
| **Python**             | Main programming language                 |
| **TensorFlow / Keras** | CNN development and model training        |
| **NumPy**              | Numerical and image-array operations      |
| **OpenCV**             | Image processing                          |
| **Pillow (PIL)**       | Image loading and manipulation            |
| **Scikit-learn**       | Machine learning utilities and evaluation |
| **Matplotlib**         | Data visualization                        |
| **Streamlit**          | Web application interface                 |

---

#  Project Structure

A typical project structure is:

```text
Project-Exhibition-01/
│
├── app.py
├── requirements.txt
│
├── model/
│   └── trained_model
│
├── dataset/
│   ├── benign/
│   └── malignant/
│
├── notebooks/
│   └── model_training.ipynb
│
├── images/
│   └── screenshots
│
└── README.md
```

> The exact folder structure may vary depending on the final project implementation.

---

#  Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
```

Navigate to the project directory:

```bash
cd Project-Exhibition-01
```

---

## 2. Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

#  Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL.

Open the URL in your browser to access the application.

---

#  Model Prediction

The prediction pipeline can be summarized as:

```text
Input Image
     ↓
Image Validation
     ↓
Image Resize
     ↓
Normalization
     ↓
CNN Model
     ↓
Prediction Probability
     ↓
Classification
     ↓
Benign / Malignant
```

The final output contains:

```text
Prediction: Benign
Confidence: XX.XX%
```

or

```text
Prediction: Malignant
Confidence: XX.XX%
```

---

# ✨ Key Features

###  Image Upload

Users can upload histopathology images directly through the web interface.

### Input Validation

The application validates the uploaded image before sending it to the model.

### CNN-Based Classification

A trained Convolutional Neural Network is used for image classification.

###  Confidence Score

The system provides the model's confidence along with the predicted class.

###  Streamlit Interface

A simple web interface makes the model accessible without requiring users to interact directly with Python code.

###  End-to-End Workflow

The project demonstrates the complete pipeline:

```text
Image → Validation → Preprocessing → Model → Prediction
```

---

#  Model Evaluation

The trained model can be evaluated using standard classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Example:

```text
              Predicted
              Benign   Malignant
Actual
Benign          TP        FN
Malignant       FP        TP
```

> Add the actual evaluation results here once the final trained model metrics are available.

For example:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | XX.XX% |
| Precision | XX.XX% |
| Recall    | XX.XX% |
| F1-Score  | XX.XX% |

---

#  Team & Contributions

## Author

### **Shwetank Vaibhav**

---

## Team Members

* **Pranshu Gupta**
* **Ayush Shukla**
* **Debasish Kumar Sahoo**
* **Manvendra Kumar**
* **Piyush Kumar Dash**

The project was developed collaboratively, with team members contributing to different areas including:

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

#  Limitations

The current project has several limitations:

1. The model's performance depends heavily on the quality and diversity of the training dataset.
2. Histopathology images may vary in staining, magnification, and acquisition conditions.
3. A machine learning prediction should not be treated as a confirmed medical diagnosis.
4. Model confidence does not necessarily represent clinical certainty.
5. The system is intended primarily as an educational/research demonstration.

---

#  Future Scope

The project can be further improved by:

### Advanced Deep Learning Models

Experimenting with architectures such as:

* ResNet
* EfficientNet
* DenseNet
* Vision Transformers

###  Improved Dataset

Training with a larger and more diverse dataset can improve generalization.

###  Explainable AI

Techniques such as **Grad-CAM** can be added to visualize the regions of an image that influenced the model's prediction.

###  Improved Application

The Streamlit application could be enhanced with:

* Prediction history
* Batch image processing
* Interactive visualizations
* Model performance dashboard

###  Deployment

The application could be deployed to a cloud platform for easier accessibility and demonstration.

---

#  Conclusion

**Project Exhibition 01** demonstrates how Artificial Intelligence and Deep Learning can be applied to breast histopathology image classification.

The system combines:

```text
Image Validation
       +
Image Preprocessing
       +
CNN Classification
       +
Confidence Estimation
       +
Streamlit Interface
```

to create a simple end-to-end machine learning application capable of classifying histopathology images as **Benign** or **Malignant**.

The project provides an educational demonstration of how computer vision and deep learning can be used in medical image analysis while emphasizing that such predictions should not replace professional clinical evaluation.

---

##  Project Highlights

> **AI/ML Project**
> **Domain:** Medical Image Analysis
> **Task:** Binary Image Classification
> **Model:** Convolutional Neural Network (CNN)
> **Classes:** Benign / Malignant
> **Language:** Python
> **Interface:** Streamlit

---

##  Disclaimer

This project is intended **only for educational and research purposes**. The predictions generated by the model should not be used as a substitute for professional medical advice, diagnosis, or treatment.


