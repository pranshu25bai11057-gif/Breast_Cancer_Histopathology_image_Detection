# Project Exhibition 01

## Breast Cancer Detection Using Histopathology and Machine Learning

Project Exhibition 01 is an AI/ML project developed to classify breast histopathology images as **Benign** or **Malignant**.

The system performs image validation before prediction. Valid images are preprocessed and passed to a single CNN model for classification.

---

## How It Works

The system follows a simple pipeline:

```text
User Uploads Image
        ↓
Image Validation / Segregation
        ↓
Is the image suitable for histopathology analysis?
        ↓
   ┌───────────────┐
   │               │
  NO              YES
   │               │
   ↓               ↓
Invalid Image   Preprocessing
                   ↓
                CNN Model
                   ↓
             Benign / Malignant
                   ↓
          Prediction + Confidence
                   ↓
               Final Result
```

### Step-by-step

1. **Image Upload**  
   The user uploads a histopathology image through the application.

2. **Image Validation**  
   The system checks whether the uploaded file is a usable image and suitable for the histopathology workflow. Clearly invalid or unrelated inputs are rejected before prediction.

3. **Preprocessing**  
   A valid image is converted to the required format, resized to the model's input size, and normalized using the same preprocessing approach used during training.

4. **CNN Prediction**  
   The processed image is passed to the trained CNN model.

5. **Result**  
   The application displays either **Benign** or **Malignant**, together with the model's output confidence.

---

## Technology Used

The project is primarily built with **Python**.

| Technology | Purpose |
|---|---|
| Python | Main development language |
| TensorFlow / Keras | CNN development and training |
| NumPy | Image arrays and numerical operations |
| OpenCV | Image processing |
| Pillow | Image loading and handling |
| Scikit-learn | Machine learning utilities |
| Matplotlib | Visualizations |
| Streamlit | Web interface for the application |

The implementation is kept focused on the core image-classification workflow without unnecessary backend infrastructure.

---

## Deployment / Running the Project

The project can be run locally using Python.

### 1. Clone the repository

```bash
git clone <repository-url>
cd Project-Exhibition-01
```

### 2. Create a Python environment

```bash
python -m venv venv
```

Activate the environment using the appropriate command for your operating system.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the project

Make sure the required project files and trained model are available in their expected locations.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in the browser and provide the image-upload interface.

---

## Authors & Contributions

### Author

**Shwetank Vaibhav**

### Contributions

- **Pranshu Gupta**
- **Ayush Shukla**
- **Debasish Kumar Sahoo**
- **Manvendra Kumar**
- **Piyush Kumar Das**

The project was developed collaboratively, with team members contributing to areas such as research, machine learning, application development, testing, and presentation.

---

## Conclusion

Project Exhibition 01 demonstrates a simple and explainable approach to breast histopathology image classification using machine learning. By validating images before prediction and using a focused CNN-based workflow, the project provides a practical demonstration of an end-to-end AI/ML application.

---
