import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = (128, 128)
CLASS_NAMES = ['benign', 'invalid', 'malignant']
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "breast_cancer_model.keras")

_CACHED_MODEL = None

def load_trained_model(model_path=DEFAULT_MODEL_PATH):
    """
    Loads and caches the trained Keras CNN model.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'. Please train the model first.")
        _CACHED_MODEL = tf.keras.models.load_model(model_path)
    return _CACHED_MODEL

def load_image(image_input):
    """
    Loads and validates an image from a file path, bytes, io.BytesIO, or PIL Image.
    Returns RGB PIL Image or raises ValueError/IOError if corrupted/unsupported.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image path does not exist: {image_input}")
        img = Image.open(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, 'read'):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        raise ValueError("Unsupported image input type.")
        
    # Verify image integrity and convert to RGB
    img.verify()
    # Re-open after verify() because verify() can reset file pointer
    if isinstance(image_input, (str, bytes, bytearray)) or hasattr(image_input, 'read'):
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
        elif hasattr(image_input, 'seek'):
            image_input.seek(0)
            img = Image.open(image_input)
            
    img = img.convert("RGB")
    return img

def prepare_image(image_input, target_size=IMAGE_SIZE):
    """
    Unified preprocessing:
    1. Load / Convert to RGB
    2. Resize to target size (128, 128)
    3. Normalize pixel values to [0, 1]
    4. Expand dimensions for batch input (1, 128, 128, 3)
    """
    img = load_image(image_input)
    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def predict_cancer(image_input, model=None, model_path=DEFAULT_MODEL_PATH):
    """
    Main prediction pipeline:
    Validates input -> Preprocesses -> CNN Inference -> Outputs Result & Confidence
    """
    try:
        # Step 1: Validate and Preprocess
        processed_tensor = prepare_image(image_input)
    except Exception as e:
        return {
            "status": "error",
            "prediction": "Invalid",
            "is_valid_histopathology": False,
            "confidence": 0.0,
            "probabilities": {},
            "message": f"Invalid or corrupted image file: {str(e)}"
        }

    # Step 2: Load Model
    try:
        if model is None:
            model = load_trained_model(model_path)
    except Exception as e:
        return {
            "status": "error",
            "prediction": "Error",
            "is_valid_histopathology": False,
            "confidence": 0.0,
            "probabilities": {},
            "message": f"Failed to load model: {str(e)}"
        }

    # Step 3: Run Inference
    raw_probs = model.predict(processed_tensor, verbose=0)[0]
    
    # Map probabilities to class names: ['benign', 'invalid', 'malignant']
    prob_dict = {
        CLASS_NAMES[i]: float(raw_probs[i])
        for i in range(len(CLASS_NAMES))
    }
    
    predicted_idx = int(np.argmax(raw_probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence_score = float(raw_probs[predicted_idx]) * 100.0
    
    # Step 4: Format output
    if predicted_class == 'invalid':
        return {
            "status": "success",
            "prediction": "Invalid",
            "is_valid_histopathology": False,
            "confidence": round(confidence_score, 2),
            "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
            "message": "Invalid image — please upload a suitable breast histopathology image."
        }
    else:
        result_label = "Benign" if predicted_class == "benign" else "Malignant"
        return {
            "status": "success",
            "prediction": result_label,
            "is_valid_histopathology": True,
            "confidence": round(confidence_score, 2),
            "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
            "message": f"Prediction: {result_label} with {confidence_score:.2f}% confidence."
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Testing prediction on: {test_file}")
        res = predict_cancer(test_file)
        print(res)
    else:
        print("Usage: python predict.py <path_to_image>")
