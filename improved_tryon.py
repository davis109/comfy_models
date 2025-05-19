import os
import sys
import argparse
import datetime
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import cv2

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def extract_foreground(image_path):
    """Extract the foreground from the clothing image using OpenCV"""
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # If image has alpha channel, use it for transparency
    if img.shape[2] == 4:
        # Get alpha channel
        alpha = img[:, :, 3]
        # Create mask from alpha
        mask = alpha > 0
        return img, mask
    
    # Otherwise, use color-based segmentation
    # Convert to RGB for display
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create a binary mask for the main object
    # Adjust these thresholds based on the clothing color
    lower = np.array([0, 0, 0])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Find the largest contour (assuming it's the clothing)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [max_contour], 0, 255, -1)
    
    # Apply some morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Convert mask to boolean
    mask_bool = mask > 0
    
    return rgb_img, mask_bool

def detect_person(image_path):
    """Detect key body points to help with clothing alignment"""
    # Load image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # For simplicity, we'll estimate:
    # - Shoulder line at ~20% down from the top
    # - Waist at ~45% down
    shoulder_y = int(height * 0.20)
    waist_y = int(height * 0.45)
    
    # Shoulder width is approximately 50% of image width
    shoulder_width = int(width * 0.5)
    shoulder_left = (width - shoulder_width) // 2
    shoulder_right = shoulder_left + shoulder_width
    
    # Return key points
    return {
        'shoulder_line': shoulder_y,
        'shoulder_left': shoulder_left,
        'shoulder_right': shoulder_right,
        'waist_line': waist_y,
        'height': height,
        'width': width
    }

def enhanced_tryon(person_img_path, cloth_img_path, output_path):
    """Perform enhanced clothing try-on with better blending"""
    try:
        # Load person image
        person_img = Image.open(person_img_path).convert('RGB')
        
        # Extract cloth and its mask
        cloth_array, cloth_mask = extract_foreground(cloth_img_path)
        
        # Convert numpy array to PIL image
        cloth_img = Image.fromarray(cloth_array)
        
        # Detect body keypoints
        body_points = detect_person(person_img_path)
        
        # Resize cloth to fit the person's upper body
        cloth_width = body_points['shoulder_right'] - body_points['shoulder_left']
        # Calculate height while maintaining aspect ratio
        orig_width, orig_height = cloth_img.size
        cloth_height = int(orig_height * (cloth_width / orig_width))
        
        # Resize cloth
        cloth_img = cloth_img.resize((cloth_width, cloth_height), Image.LANCZOS)
        
        # Calculate paste position
        paste_x = body_points['shoulder_left']
        paste_y = body_points['shoulder_line'] - int(cloth_height * 0.2)  # Place shirt slightly above shoulder line
        
        # Create a result image
        result_img = person_img.copy()
        
        # Create a mask for the cloth (from boolean mask to PIL mask)
        pil_mask = Image.fromarray(cloth_mask.astype(np.uint8) * 255).resize(
            (cloth_width, cloth_height), Image.LANCZOS)
        
        # Apply slight Gaussian blur to the mask edges for smoother blending
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Convert to RGBA for better compositing
        if cloth_img.mode != 'RGBA':
            cloth_img = cloth_img.convert('RGBA')
        
        # Paste the cloth onto the person using the mask
        result_img.paste(cloth_img, (paste_x, paste_y), pil_mask)
        
        # Apply slight blur at the boundaries for better blending
        result_img = result_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Save result
        result_img.save(output_path)
        print(f"Enhanced try-on created and saved to {output_path}")
        return result_img
    except Exception as e:
        print(f"Error in enhanced try-on: {e}")
        return None

def create_tryon_visualization(person_img_path, cloth_img_path, result_img_path):
    """Create a side-by-side visualization of inputs and output"""
    print("Creating visualization...")
    
    # Output path with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = f"results/final_presentation_{timestamp}.png"
    ensure_dir("results")
    
    # Load images
    person_img = Image.open(person_img_path).convert('RGB')
    cloth_img = Image.open(cloth_img_path).convert('RGBA')
    result_img = Image.open(result_img_path).convert('RGB')
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Original person image
    axes[0].imshow(np.array(person_img))
    axes[0].set_title("Original Person", fontsize=14)
    axes[0].axis('off')
    
    # Original clothing image
    # Create white background
    white_bg = Image.new('RGBA', cloth_img.size, (255, 255, 255, 255))
    # Paste cloth image onto white background
    cloth_display = Image.alpha_composite(white_bg, cloth_img)
    axes[1].imshow(np.array(cloth_display.convert('RGB')))
    axes[1].set_title("Clothing Item", fontsize=14)
    axes[1].axis('off')
    
    # Result image
    axes[2].imshow(np.array(result_img))
    axes[2].set_title("Virtual Try-On Result", fontsize=14)
    axes[2].axis('off')
    
    # Add main title
    plt.suptitle("Virtual Try-On Demo", fontsize=18, y=0.95)
    
    # Tight layout and adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, top=0.85)
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description="Virtual Try-On Application")
    parser.add_argument("--person", type=str, required=True, help="Path to person image")
    parser.add_argument("--cloth", type=str, required=True, help="Path to clothing image")
    parser.add_argument("--output", type=str, default="results/enhanced_tryon.png", help="Output path for try-on result")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Ensure output directory exists
    ensure_dir(os.path.dirname(args.output))
    
    # Perform the enhanced try-on
    print(f"Processing person image: {args.person}")
    print(f"Processing clothing image: {args.cloth}")
    
    # Run enhanced try-on
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_path = f"results/enhanced_tryon_{timestamp}.png"
    ensure_dir("results")
    
    result_img = enhanced_tryon(args.person, args.cloth, result_path)
    if result_img:
        # Create visualization
        viz_path = create_tryon_visualization(args.person, args.cloth, result_path)
        print(f"Try-on completed successfully!")
        print(f"Result saved to: {result_path}")
        print(f"Visualization saved to: {viz_path}")
    else:
        print("Try-on process failed.")

if __name__ == "__main__":
    main() 