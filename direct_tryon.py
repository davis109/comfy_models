import os
import sys
import subprocess
from PIL import Image

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    # Set up output directory
    output_dir = "results"
    ensure_dir(output_dir)
    
    # Copy the images to the proper location
    custom_person_img = "custom_test/image/person.png"
    custom_cloth_img = "custom_test/clothes/shirt.png"
    
    # Define output path
    output_path = os.path.join(output_dir, "custom_tryon_result.png")
    
    # Run the demo script directly with our images
    cmd = [
        sys.executable, "demo.py",
        "--person_image", custom_person_img,
        "--garment_image", custom_cloth_img,
        "--output_image", output_path
    ]
    
    print("Running virtual try-on...")
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    print(process.stdout)
    
    if process.returncode != 0:
        print(f"Virtual try-on failed with return code {process.returncode}")
    else:
        print(f"Virtual try-on completed successfully!")
        print(f"Result saved to: {output_path}")
        
        # Resize the result for display
        try:
            img = Image.open(output_path)
            # Display result info
            print(f"Output image dimensions: {img.size}")
        except Exception as e:
            print(f"Error opening result image: {e}")

if __name__ == "__main__":
    main() 