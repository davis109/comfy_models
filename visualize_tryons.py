import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_tryon_results(person_img_path, cloth_img_path, results_dir="results_custom"):
    """Create a visualization of the StableVITON virtual try-on results"""
    print("Visualizing StableVITON try-on results...")
    
    # Load input images
    person_img = Image.open(person_img_path).convert('RGB')
    cloth_img = Image.open(cloth_img_path).convert('RGB')
    
    # Find result image
    result_files = []
    for ext in ['*.jpg', '*.png']:
        pattern = os.path.join(results_dir, 'pair', ext)
        result_files.extend(glob.glob(pattern))
    
    if not result_files:
        print(f"No result images found in {os.path.join(results_dir, 'pair')}")
        # Look in just the results directory
        for ext in ['*.jpg', '*.png']:
            pattern = os.path.join(results_dir, ext)
            result_files.extend(glob.glob(pattern))
    
    if not result_files:
        # Try using the dataset preparation files as a fallback
        dataset_dir = "StableVITON/data/test"
        person_path = os.path.join(dataset_dir, "image", "person_001.jpg")
        cloth_path = os.path.join(dataset_dir, "cloth", "cloth_001.jpg")
        agnostic_path = os.path.join(dataset_dir, "agnostic", "person_001.jpg")
        
        # Create a visualization anyway
        plt.figure(figsize=(15, 5))
        
        # Original person image
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(Image.open(person_path)))
        plt.title("Person Image")
        plt.axis('off')
        
        # Original clothing image
        plt.subplot(1, 3, 2)
        plt.imshow(np.array(Image.open(cloth_path)))
        plt.title("Clothing Item")
        plt.axis('off')
        
        # Agnostic image (placeholder for result)
        plt.subplot(1, 3, 3)
        plt.imshow(np.array(Image.open(agnostic_path)))
        plt.title("Prepared Input (Result Pending)")
        plt.axis('off')
        
        plt.suptitle("StableVITON Dataset Preparation (Result Pending)", fontsize=16)
        
        # Save figure
        output_path = "stableviton_visualization_prep.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Dataset preparation visualization saved to {output_path}")
        print("No try-on results found yet. Run StableVITON's inference.py script to generate results.")
        return output_path
    
    # Use the first result found
    result_img_path = result_files[0]
    result_img = Image.open(result_img_path).convert('RGB')
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Original person image
    plt.subplot(1, 3, 1)
    plt.imshow(np.array(person_img))
    plt.title("Person Image")
    plt.axis('off')
    
    # Original clothing image
    plt.subplot(1, 3, 2)
    plt.imshow(np.array(cloth_img))
    plt.title("Clothing Item")
    plt.axis('off')
    
    # Result image
    plt.subplot(1, 3, 3)
    plt.imshow(np.array(result_img))
    plt.title("StableVITON Result")
    plt.axis('off')
    
    # Add title
    plt.suptitle("StableVITON Virtual Try-On", fontsize=16)
    
    # Save figure
    output_path = "stableviton_visualization_result.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Default values
    person_img = "zz.png"
    cloth_img = "shirt.png"
    results_dir = "results_custom"
    
    # Override with command line args if provided
    if len(sys.argv) > 1:
        person_img = sys.argv[1]
    if len(sys.argv) > 2:
        cloth_img = sys.argv[2]
    if len(sys.argv) > 3:
        results_dir = sys.argv[3]
    
    # Visualize
    visualize_tryon_results(person_img, cloth_img, results_dir) 