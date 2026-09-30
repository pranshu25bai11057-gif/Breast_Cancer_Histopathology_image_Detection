import os
import re
import shutil
import random
import numpy as np
from PIL import Image, ImageDraw

def get_patient_id(file_path):
    """
    Extracts patient identifier from BreaKHis file path.
    Example path: .../SOB_M_MC_14-13418DE/100X/SOB_M_MC-14-13418DE-100-009.png
    Patient ID: 14-13418DE
    """
    filename = os.path.basename(file_path)
    # Match pattern like 14-13418DE or 14-9988 or 14-22549AB
    match = re.search(r'(\d{2}-\d+[A-Z]*)', filename)
    if match:
        return match.group(1)
    
    # Fallback to parent directory name
    parent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    return parent_dir

def generate_invalid_images(output_dir, count=300):
    """
    Generates diverse non-histopathology synthetic images for the 'invalid' class.
    Includes gradients, geometric patterns, noise, text, and synthetic objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    
    for i in range(count):
        img_type = i % 6
        width, height = 200, 200
        
        if img_type == 0:
            # Random color gradient
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            c1 = np.random.randint(0, 255, 3)
            c2 = np.random.randint(0, 255, 3)
            for y in range(height):
                alpha = y / height
                img_array[y, :] = (1 - alpha) * c1 + alpha * c2
            img = Image.fromarray(img_array)
            
        elif img_type == 1:
            # Checkerboard / geometric grid
            img = Image.new('RGB', (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            grid_size = random.choice([10, 20, 25, 40])
            c1 = tuple(np.random.randint(0, 255, 3).tolist())
            c2 = tuple(np.random.randint(0, 255, 3).tolist())
            for x in range(0, width, grid_size):
                for y in range(0, height, grid_size):
                    if (x // grid_size + y // grid_size) % 2 == 0:
                        draw.rectangle([x, y, x + grid_size, y + grid_size], fill=c1)
                    else:
                        draw.rectangle([x, y, x + grid_size, y + grid_size], fill=c2)
                        
        elif img_type == 2:
            # Geometric shapes (circles, rectangles, triangles)
            img = Image.new('RGB', (width, height), color=tuple(np.random.randint(200, 255, 3).tolist()))
            draw = ImageDraw.Draw(img)
            for _ in range(random.randint(5, 15)):
                x1, y1 = random.randint(0, width-50), random.randint(0, height-50)
                x2, y2 = x1 + random.randint(20, 60), y1 + random.randint(20, 60)
                color = tuple(np.random.randint(0, 220, 3).tolist())
                shape_choice = random.choice(['rect', 'ellipse', 'line'])
                if shape_choice == 'rect':
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0))
                elif shape_choice == 'ellipse':
                    draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0))
                else:
                    draw.line([x1, y1, x2, y2], fill=color, width=3)
                    
        elif img_type == 3:
            # Structured noise texture (non-histological)
            noise = np.random.normal(128, 40, (height, width, 3)).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(noise)
            
        elif img_type == 4:
            # Synthetic text / document / diagram
            img = Image.new('RGB', (width, height), color=(250, 250, 250))
            draw = ImageDraw.Draw(img)
            for y_line in range(20, height - 20, 15):
                draw.line([20, y_line, random.randint(80, width - 20), y_line], fill=(50, 50, 50), width=2)
            # Add a colored box or stamp
            draw.rectangle([120, 120, 180, 180], outline=(200, 30, 30), width=3)
            
        else:
            # High-contrast natural colors (e.g. green grass / blue sky mockups)
            img_array = np.zeros((height, width, 3), dtype=np.uint8)
            # Blue sky on top
            img_array[:height//2, :] = [135, 206, 235] + np.random.randint(-10, 10, (height//2, width, 3))
            # Green grass on bottom
            img_array[height//2:, :] = [34, 139, 34] + np.random.randint(-10, 10, (height - height//2, width, 3))
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
        img.save(os.path.join(output_dir, f"invalid_{i:04d}.png"))

def prepare_dataset(raw_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Groups BreaKHis images by patient ID and creates patient-level train/validation/test splits.
    """
    random.seed(seed)
    
    classes = ['benign', 'malignant']
    base_slides_dir = os.path.join(raw_dir, 'histology_slides', 'breast')
    
    split_counts = {'train': {}, 'validation': {}, 'test': {}}
    
    for cls in classes:
        cls_dir = os.path.join(base_slides_dir, cls)
        patient_images = {}
        
        for root, _, files in os.walk(cls_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    full_path = os.path.join(root, file)
                    pid = get_patient_id(full_path)
                    if pid not in patient_images:
                        patient_images[pid] = []
                    patient_images[pid].append(full_path)
                    
        patient_ids = list(patient_images.keys())
        patient_ids.sort()
        random.shuffle(patient_ids)
        
        n_total = len(patient_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_patients = set(patient_ids[:n_train])
        val_patients = set(patient_ids[n_train:n_train + n_val])
        test_patients = set(patient_ids[n_train + n_val:])
        
        print(f"[{cls.upper()}] Total Patients: {n_total} -> Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
        
        # Copy files to destination splits
        for split_name, split_pats in [('train', train_patients), ('validation', val_patients), ('test', test_patients)]:
            dst_cls_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(dst_cls_dir, exist_ok=True)
            
            count = 0
            for pid in split_pats:
                for img_path in patient_images[pid]:
                    filename = os.path.basename(img_path)
                    dst_path = os.path.join(dst_cls_dir, filename)
                    if not os.path.exists(dst_path):
                        shutil.copy2(img_path, dst_path)
                    count += 1
            split_counts[split_name][cls] = count
            
    # Generate Invalid images for all splits
    print("[INVALID] Generating non-histopathology images for Invalid class...")
    generate_invalid_images(os.path.join(output_dir, 'train', 'invalid'), count=350)
    generate_invalid_images(os.path.join(output_dir, 'validation', 'invalid'), count=75)
    generate_invalid_images(os.path.join(output_dir, 'test', 'invalid'), count=75)
    
    split_counts['train']['invalid'] = 350
    split_counts['validation']['invalid'] = 75
    split_counts['test']['invalid'] = 75
    
    print("\nDataset Preparation Completed Successfully!")
    print("---------------------------------------------")
    for split in ['train', 'validation', 'test']:
        print(f"Split [{split}]:")
        for c in ['benign', 'malignant', 'invalid']:
            print(f"  - {c}: {split_counts[split].get(c, 0)} images")
    print("---------------------------------------------")

if __name__ == "__main__":
    raw_dataset_dir = "BreaKHis_v1"
    prepared_data_dir = "data"
    prepare_dataset(raw_dataset_dir, prepared_data_dir)
