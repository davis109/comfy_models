import os
import subprocess
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
from datetime import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def create_mask_from_image(image_path, output_path):
    """Create a simple mask for the clothing item"""
    try:
        # Open the image
        img = Image.open(image_path).convert('RGBA')
        width, height = img.size
        
        # Create a white mask where the clothing is (non-transparent areas)
        mask = Image.new('L', (width, height), 0)
        for y in range(height):
            for x in range(width):
                r, g, b, a = img.getpixel((x, y))
                if a > 100:  # If not very transparent
                    mask.putpixel((x, y), 255)
        
        # Save the mask
        mask.save(output_path)
        return True
    except Exception as e:
        print(f"Error creating mask: {e}")
        return False

def prepare_images():
    """Prepare the images for the model"""
    # Create the necessary directories
    ensure_dir("custom_output")
    ensure_dir("StableVITON/data/custom/image")
    ensure_dir("StableVITON/data/custom/cloth")
    ensure_dir("StableVITON/data/custom/cloth_mask")
    
    # Copy images to the StableVITON data directory
    src_person = "custom_test/image/person.png"
    src_cloth = "custom_test/clothes/shirt.png"
    
    dst_person = "StableVITON/data/custom/image/person.png"
    dst_cloth = "StableVITON/data/custom/cloth/shirt.png"
    dst_cloth_mask = "StableVITON/data/custom/cloth_mask/shirt.png"
    
    # Copy files
    copy_cmd1 = f'Copy-Item -Path "{src_person}" -Destination "{dst_person}" -Force'
    copy_cmd2 = f'Copy-Item -Path "{src_cloth}" -Destination "{dst_cloth}" -Force'
    
    subprocess.run(["powershell", "-Command", copy_cmd1], check=False)
    subprocess.run(["powershell", "-Command", copy_cmd2], check=False)
    
    # Create mask for the clothing
    create_mask_from_image(src_cloth, dst_cloth_mask)
    
    # Create a test pair file
    with open("StableVITON/data/custom/test_pairs.txt", "w") as f:
        f.write("person.png shirt.png")

def main():
    print("Preparing images for virtual try-on...")
    prepare_images()
    
    # Configure to use VAE fine-tuning
    print("Configuring to use VAE fine-tuning...")
    try:
        vae_cmd = [
            "python", "StableVITON/use_vae.py",
            "--config_file", "StableVITON/configs/VITONHD.yaml",
            "--vae_ckpt", "VITONHD_VAE_finetuning.ckpt"
        ]
        subprocess.run(vae_cmd, check=True)
    except subprocess.CalledProcessError:
        print("Warning: Failed to configure VAE. Continuing with default settings.")
    
    # Run inference
    print("Running virtual try-on with StableVITON...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"custom_output_{timestamp}"
    ensure_dir(output_dir)
    
    inference_cmd = [
        "python", "StableVITON/inference.py",
        "--config_path", "StableVITON/configs/VITONHD.yaml",
        "--model_load_path", "VITONHD_PBE_POSE.ckpt",
        "--batch_size", "1",
        "--data_root_dir", "StableVITON/data",
        "--datamode", "custom",  # Use our custom dataset
        "--save_dir", output_dir,
        "--denoise_steps", "50",  # More steps for better quality
        "--img_H", "512",
        "--img_W", "384"
    ]
    
    # Run the command
    process = subprocess.Popen(inference_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    
    if process.returncode != 0:
        print(f"Virtual try-on failed with return code {process.returncode}")
    else:
        print(f"Virtual try-on completed successfully. Results saved to {output_dir}")
        
        # Copy the result to the main directory for easy access
        result_files = os.listdir(output_dir)
        if result_files:
            result_file = os.path.join(output_dir, result_files[0])
            copy_cmd = f'Copy-Item -Path "{result_file}" -Destination "custom_result.png" -Force'
            subprocess.run(["powershell", "-Command", copy_cmd], check=False)
            print("Result also saved as custom_result.png in the main directory")

if __name__ == "__main__":
    main() 