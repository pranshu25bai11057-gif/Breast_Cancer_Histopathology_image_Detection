import os
import glob
import numpy as np
from PIL import Image
import predict

def run_all_tests():
    print("==================================================")
    print("       RUNNING SYSTEM VERIFICATION TESTS          ")
    print("==================================================")
    
    # Check 1: Model file exists
    model_path = predict.DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"FAILED: Model file not found at {model_path}. Train the model first.")
        return False
    print("✓ Model file exists.")
    
    # Load model
    try:
        model = predict.load_trained_model(model_path)
        print("✓ Model loaded successfully.")
    except Exception as e:
        print(f"FAILED: Could not load model: {e}")
        return False

    # Check 2: Test on Benign sample
    benign_samples = glob.glob("data/test/benign/*.png")
    if benign_samples:
        test_sample = benign_samples[0]
        res = predict.predict_cancer(test_sample, model=model)
        assert res["status"] == "success", f"Benign test status failed: {res}"
        assert res["is_valid_histopathology"] is True, f"Expected valid histopathology: {res}"
        assert 0.0 <= res["confidence"] <= 100.0, f"Confidence out of bounds: {res['confidence']}"
        print(f"✓ Benign test image processed: Pred={res['prediction']}, Conf={res['confidence']}%")
    else:
        print("! Warning: No benign test images found in data/test/benign")

    # Check 3: Test on Malignant sample
    malignant_samples = glob.glob("data/test/malignant/*.png")
    if malignant_samples:
        test_sample = malignant_samples[0]
        res = predict.predict_cancer(test_sample, model=model)
        assert res["status"] == "success", f"Malignant test status failed: {res}"
        assert res["is_valid_histopathology"] is True, f"Expected valid histopathology: {res}"
        assert 0.0 <= res["confidence"] <= 100.0, f"Confidence out of bounds: {res['confidence']}"
        print(f"✓ Malignant test image processed: Pred={res['prediction']}, Conf={res['confidence']}%")
    else:
        print("! Warning: No malignant test images found in data/test/malignant")

    # Check 4: Test on Invalid non-histopathology sample
    invalid_samples = glob.glob("data/test/invalid/*.png")
    if invalid_samples:
        test_sample = invalid_samples[0]
        res = predict.predict_cancer(test_sample, model=model)
        assert res["status"] == "success", f"Invalid test status failed: {res}"
        assert res["prediction"] == "Invalid", f"Expected Invalid prediction: {res}"
        assert res["is_valid_histopathology"] is False, f"Expected is_valid_histopathology=False: {res}"
        print(f"✓ Invalid image properly rejected: Msg='{res['message']}', Conf={res['confidence']}%")
    else:
        print("! Warning: No invalid test images found in data/test/invalid")

    # Check 5: Corrupted file handling
    corrupted_bytes = b"NOT_A_REAL_IMAGE_FILE_JUST_RANDOM_CORRUPT_BYTES_12345"
    res_corrupt = predict.predict_cancer(corrupted_bytes, model=model)
    assert res_corrupt["status"] == "error", f"Expected error status for corrupted file: {res_corrupt}"
    assert res_corrupt["prediction"] == "Invalid", f"Expected Invalid for corrupted file: {res_corrupt}"
    print("✓ Corrupted file safely caught by validation gate.")

    print("\n==================================================")
    print("       ALL SYSTEM VERIFICATION TESTS PASSED       ")
    print("==================================================")
    return True

if __name__ == "__main__":
    run_all_tests()
