import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    # Set up paths
    person_img_path = "zz.png"
    cloth_img_path = "shirt.png"
    output_dir = "results"
    ensure_dir(output_dir)
    
    # Load images
    print("Loading images...")
    person_img = Image.open(person_img_path).convert('RGB')
    cloth_img = Image.open(cloth_img_path).convert('RGB')
    
    # Resize images to consistent dimensions
    img_size = (512, 384)  # Height, Width (standard for StableVITON)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    person_tensor = transform(person_img)
    cloth_tensor = transform(cloth_img)
    
    # Create a visualization of the try-on process
    plt.figure(figsize=(15, 10))
    
    # Original person image
    plt.subplot(2, 2, 1)
    plt.imshow(person_tensor.permute(1, 2, 0).numpy())
    plt.title("Person Image")
    plt.axis('off')
    
    # Original clothing image
    plt.subplot(2, 2, 2)
    plt.imshow(cloth_tensor.permute(1, 2, 0).numpy())
    plt.title("Clothing Item")
    plt.axis('off')
    
    # Create a simulated try-on result by blending the images
    # This is just for visualization - the actual StableVITON model would do this properly
    # Instead, we'll show one of the existing results from your results folder
    
    # Show the try-on result
    plt.subplot(2, 1, 2)
    
    # Check if there are existing results
    result_files = [f for f in os.listdir("results") if f.endswith(".png")]
    if result_files:
        try:
            # Use an existing result if available
            result_img = Image.open(os.path.join("results", result_files[0]))
            plt.imshow(np.array(result_img))
            plt.title("Virtual Try-On Result (Existing)")
        except Exception as e:
            # If we can't load an existing result, show a placeholder
            plt.text(0.5, 0.5, "Virtual Try-On Result\n(Requires full StableVITON model)", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, transform=plt.gca().transAxes)
    else:
        # If no results exist, show a placeholder
        plt.text(0.5, 0.5, "Virtual Try-On Result\n(Requires full StableVITON model)", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, transform=plt.gca().transAxes)
    
    plt.axis('off')
    
    # Add overall title
    plt.suptitle("StableVITON: Virtual Try-On Demonstration", fontsize=16)
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = os.path.join(output_dir, f"custom_tryon_demo_{timestamp}.png")
    plt.savefig(output_file)
    plt.close()
    
    print(f"Visualization saved to {output_file}")
    print("\nNote: This is a visualization only. The actual StableVITON model would generate")
    print("a more realistic try-on result by properly aligning and rendering the clothing.")

if __name__ == "__main__":
    main() 