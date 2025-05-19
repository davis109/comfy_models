import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageOps

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def create_agnostic_image(person_img_path, output_path):
    """Create an agnostic image (remove clothing region)"""
    # Load the person image
    img = Image.open(person_img_path).convert('RGB')
    img_array = np.array(img)
    
    # Estimate clothing region (upper body)
    height, width = img_array.shape[:2]
    
    # Create a simple mask for the upper body area
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define upper body region (simple rectangle)
    # Adjust these values based on your specific images
    upper_y = int(height * 0.15)  # Start at 15% from top
    lower_y = int(height * 0.45)  # End at 45% from top
    left_x = int(width * 0.25)    # 25% from left
    right_x = int(width * 0.75)   # 25% from right
    
    # Create the mask
    mask[upper_y:lower_y, left_x:right_x] = 255
    
    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(img_array, (25, 25), 0)
    
    # Replace the masked area with the blurred version
    img_array_copy = img_array.copy()
    img_array_copy[mask > 0] = blurred[mask > 0]
    
    # Save the agnostic image
    agnostic_img = Image.fromarray(img_array_copy)
    agnostic_img.save(output_path)
    
    # Also create and return the mask
    mask_img = Image.fromarray(mask)
    return mask_img

def prepare_full_dataset(person_img_path, cloth_img_path):
    """Prepare a complete dataset structure for StableVITON"""
    print("Preparing complete StableVITON dataset structure...")
    
    # Base directories
    data_root = "StableVITON/data"
    data_test = os.path.join(data_root, "test")
    
    # Required directories
    test_dirs = [
        os.path.join(data_test, "image"),
        os.path.join(data_test, "cloth"),
        os.path.join(data_test, "cloth-mask"),
        os.path.join(data_test, "agnostic"),
        os.path.join(data_test, "agnostic-mask"),
        os.path.join(data_test, "image-densepose")
    ]
    
    for dir_path in test_dirs:
        ensure_dir(dir_path)
    
    # Image file names
    person_filename = "person_001.jpg"
    cloth_filename = "cloth_001.jpg"
    
    # Destination paths
    dst_person = os.path.join(data_test, "image", person_filename)
    dst_cloth = os.path.join(data_test, "cloth", cloth_filename)
    dst_cloth_mask = os.path.join(data_test, "cloth-mask", cloth_filename)
    dst_agnostic = os.path.join(data_test, "agnostic", person_filename)
    dst_agnostic_mask = os.path.join(data_test, "agnostic-mask", person_filename)
    dst_densepose = os.path.join(data_test, "image-densepose", person_filename)
    
    try:
        # Process person image
        person_img = Image.open(person_img_path)
        person_img = person_img.convert('RGB')
        person_img = person_img.resize((384, 512))  # Standard size for StableVITON
        person_img.save(dst_person)
        print(f"Saved person image to {dst_person}")
        
        # Process cloth image
        cloth_img = Image.open(cloth_img_path)
        cloth_img = cloth_img.convert('RGB')
        cloth_img = cloth_img.resize((384, 512))  # Standard size
        cloth_img.save(dst_cloth)
        print(f"Saved cloth image to {dst_cloth}")
        
        # Create cloth mask
        if 'A' in cloth_img.getbands():  # Has alpha channel
            r, g, b, a = cloth_img.split()
            mask = a.point(lambda i: 255 if i > 0 else 0)
        else:
            # Create a simple white mask (assuming white background)
            img_array = np.array(cloth_img)
            # Simple thresholding - assuming white background
            mask_array = np.all(img_array > 240, axis=2)
            mask_array = ~mask_array  # Invert the mask
            mask = Image.fromarray(mask_array.astype(np.uint8) * 255)
        
        mask = mask.convert('RGB')
        mask.save(dst_cloth_mask)
        print(f"Saved cloth mask to {dst_cloth_mask}")
        
        # Create and save agnostic image
        agnostic_mask = create_agnostic_image(dst_person, dst_agnostic)
        agnostic_mask.save(dst_agnostic_mask)
        print(f"Saved agnostic image to {dst_agnostic}")
        print(f"Saved agnostic mask to {dst_agnostic_mask}")
        
        # Create a dummy densepose visualization
        # This is normally created by a human parsing model, but we'll create a simple version
        densepose_img = Image.new('RGB', (384, 512), (128, 128, 128))
        densepose_img.save(dst_densepose)
        print(f"Saved dummy densepose image to {dst_densepose}")
        
        # Create pairs.txt file
        pairs_path = os.path.join(data_test, "test_pairs.txt")
        with open(pairs_path, "w") as f:
            f.write(f"{person_filename} {cloth_filename}")
        print(f"Created test_pairs.txt file")
        
        print("Dataset preparation complete.")
        return True
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return False

def create_cloth_mask(cloth_img_path, output_path):
    """Create a cloth mask from the cloth image"""
    try:
        # Load the cloth image
        cloth_img = Image.open(cloth_img_path)
        
        # If image has alpha channel, use it for the mask
        if cloth_img.mode == 'RGBA':
            r, g, b, a = cloth_img.split()
            mask = a.point(lambda i: 255 if i > 0 else 0)
        else:
            # Create a simple white mask (assuming white background)
            img_array = np.array(cloth_img.convert('RGB'))
            # Simple thresholding - assuming white background
            mask_array = np.all(img_array > 240, axis=2)
            mask_array = ~mask_array  # Invert the mask
            mask = Image.fromarray(mask_array.astype(np.uint8) * 255)
        
        # Save the mask
        mask = mask.convert('RGB')
        mask.save(output_path)
        print(f"Saved cloth mask to {output_path}")
        return True
    except Exception as e:
        print(f"Error creating cloth mask: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python stableviton_dataset_prep.py <person_image> <cloth_image>")
        sys.exit(1)
    
    person_img_path = sys.argv[1]
    cloth_img_path = sys.argv[2]
    
    prepare_full_dataset(person_img_path, cloth_img_path) 