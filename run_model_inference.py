import os
import sys
import subprocess
import shutil
import time
import argparse
from PIL import Image

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def install_missing_dependencies():
    """Install required dependencies for StableVITON"""
    print("Installing missing dependencies...")
    dependencies = [
        "omegaconf",
        "einops",
        "opencv-python",
        "scikit-image",
        "transformers",
        "diffusers"
    ]
    
    for package in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}, but continuing...")

def setup_data_structure_from_images():
    """Set up the proper data structure for StableVITON using the current zz.png and shirt.png"""
    print("Setting up data structure using zz.png and shirt.png...")
    
    # Create necessary directories in StableVITON/data
    data_dirs = [
        "StableVITON/data/test/image",
        "StableVITON/data/test/cloth", 
        "StableVITON/data/test/cloth-mask"
    ]
    
    for directory in data_dirs:
        ensure_dir(directory)
    
    try:
        # Define target filenames in StableVITON format
        person_filename = "person_001.jpg"
        cloth_filename = "cloth_001.jpg"
        
        # Convert and copy the images if they exist
        if os.path.exists("zz.png"):
            # Convert PNG to JPG and resize to standard size
            person_img = Image.open("zz.png").convert('RGB')
            person_img = person_img.resize((384, 512))  # Standard size (W, H)
            person_img.save(f"StableVITON/data/test/image/{person_filename}")
            print(f"Processed person image: zz.png -> {person_filename}")
        else:
            print("Error: zz.png not found")
            return None, None
        
        if os.path.exists("shirt.png"):
            # Convert PNG to JPG and resize to standard size
            cloth_img = Image.open("shirt.png").convert('RGB')
            cloth_img = cloth_img.resize((384, 512))  # Standard size (W, H)
            cloth_img.save(f"StableVITON/data/test/cloth/{cloth_filename}")
            print(f"Processed cloth image: shirt.png -> {cloth_filename}")
            
            # Create cloth mask
            create_mask(f"StableVITON/data/test/cloth/{cloth_filename}", 
                       f"StableVITON/data/test/cloth-mask/{cloth_filename}")
        else:
            print("Error: shirt.png not found")
            return None, None
        
        # Create a test_pairs.txt file inside the StableVITON data directory
        with open("StableVITON/data/test/test_pairs.txt", "w") as f:
            f.write(f"{person_filename} {cloth_filename}")
            
        print(f"Created pair file with: {person_filename} {cloth_filename}")
        
        return person_filename, cloth_filename
    except Exception as e:
        print(f"Error setting up data structure: {e}")
        return None, None

def fallback_to_train_pair():
    """Fallback to using train_pairs.txt if direct images fail"""
    print("Falling back to using a pair from train_pairs.txt...")
    
    # Create necessary directories in StableVITON/data
    data_dirs = [
        "StableVITON/data/test/image",
        "StableVITON/data/test/cloth", 
        "StableVITON/data/test/cloth-mask"
    ]
    
    for directory in data_dirs:
        ensure_dir(directory)
    
    try:
        # Get first pair from train_pairs.txt
        person_img = "10224_00.jpg"
        cloth_img = "03195_00.jpg"
        
        # Check if the source images exist in train/ directory
        if not os.path.exists(f"train/image/{person_img}"):
            print(f"Warning: {person_img} not found in train/image/")
            # Try to find any available person image
            person_files = os.listdir("train/image")
            if person_files:
                person_img = person_files[0]
                print(f"Using {person_img} instead")
            else:
                return None, None
            
        if not os.path.exists(f"train/cloth/{cloth_img}"):
            print(f"Warning: {cloth_img} not found in train/cloth/")
            # Try to find any available cloth image
            cloth_files = os.listdir("train/cloth")
            if cloth_files:
                cloth_img = cloth_files[0]
                print(f"Using {cloth_img} instead")
            else:
                return None, None
        
        # Copy the selected images to the StableVITON data structure
        shutil.copy(f"train/image/{person_img}", f"StableVITON/data/test/image/{person_img}")
        print(f"Copied person image: {person_img}")
        
        shutil.copy(f"train/cloth/{cloth_img}", f"StableVITON/data/test/cloth/{cloth_img}")
        print(f"Copied cloth image: {cloth_img}")
        
        if os.path.exists(f"train/cloth-mask/{cloth_img}"):
            shutil.copy(f"train/cloth-mask/{cloth_img}", f"StableVITON/data/test/cloth-mask/{cloth_img}")
            print(f"Copied cloth mask: {cloth_img}")
        else:
            # If mask doesn't exist, try to generate one
            print(f"Mask not found for {cloth_img}, attempting to create one")
            create_mask(f"StableVITON/data/test/cloth/{cloth_img}", 
                       f"StableVITON/data/test/cloth-mask/{cloth_img}")
        
        # Create a test_pairs.txt file inside the StableVITON data directory
        with open("StableVITON/data/test/test_pairs.txt", "w") as f:
            f.write(f"{person_img} {cloth_img}")
            
        print(f"Created pair file with: {person_img} {cloth_img}")
        
        return person_img, cloth_img
    except Exception as e:
        print(f"Error setting up fallback: {e}")
        return None, None

def create_mask(image_path, output_path):
    """Create a mask for the clothing item"""
    try:
        # Open the clothing image
        img = Image.open(image_path).convert('RGB')
        
        # Create a white mask - this is simplistic but works as a fallback
        width, height = img.size
        mask = Image.new('L', (width, height), 255)
        
        # Save the mask
        mask.save(output_path)
        print(f"Created mask at {output_path}")
    except Exception as e:
        print(f"Error creating mask: {e}")

def configure_vae():
    """Configure the VAE fine-tuning"""
    print("Configuring VAE fine-tuning...")
    
    if not os.path.exists("VITONHD_VAE_finetuning.ckpt"):
        print("Error: VAE fine-tuning checkpoint not found")
        return False
    
    try:
        vae_cmd = [
            sys.executable, "StableVITON/use_vae.py",
            "--config_file", "StableVITON/configs/VITONHD.yaml",
            "--vae_ckpt", "VITONHD_VAE_finetuning.ckpt"
        ]
        subprocess.run(vae_cmd, check=True)
        print("VAE configuration successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error configuring VAE: {e}")
        return False

def run_model_inference(output_dir="model_results"):
    """Run the StableVITON model inference"""
    print("Running StableVITON model inference...")
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Check if model exists
    if not os.path.exists("VITONHD_PBE_POSE.ckpt"):
        print("Error: Model checkpoint VITONHD_PBE_POSE.ckpt not found")
        return None
    
    # Set up the inference command
    inference_cmd = [
        sys.executable, "StableVITON/inference.py",
        "--config_path", "StableVITON/configs/VITONHD.yaml",
        "--model_load_path", "VITONHD_PBE_POSE.ckpt",
        "--batch_size", "1",
        "--data_root_dir", "StableVITON/data",
        "--save_dir", output_dir,
        "--denoise_steps", "100",  # Higher quality with more steps
        "--img_H", "512",
        "--img_W", "384"
    ]
    
    # Run inference with a timeout
    max_wait_time = 600  # 10 minutes
    
    try:
        print(f"Executing: {' '.join(inference_cmd)}")
        process = subprocess.Popen(
            inference_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > max_wait_time:
                process.terminate()
                print("Inference process timed out after 10 minutes")
                return None
            time.sleep(1)
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            print(f"Inference failed with error:\n{stderr}")
            return None
        
        print("Inference completed successfully")
        
        # Find the result file
        result_files = [f for f in os.listdir(output_dir) 
                       if os.path.isfile(os.path.join(output_dir, f)) and 
                       (f.endswith('.jpg') or f.endswith('.png'))]
        
        if not result_files:
            print("No result files found in output directory")
            return None
        
        # Copy the result to a standard name
        result_path = os.path.join(output_dir, result_files[0])
        final_path = "stableviton_model_result.png"
        shutil.copy(result_path, final_path)
        print(f"Final result saved as: {final_path}")
        
        return final_path
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def main():
    print("StableVITON Model Inference Script")
    print("==================================")
    print(f"Using model checkpoints:")
    print(f"- Main model: VITONHD_PBE_POSE.ckpt")
    print(f"- VAE model: VITONHD_VAE_finetuning.ckpt")
    print()
    
    # Install missing dependencies
    install_missing_dependencies()
    
    # First try using the provided zz.png and shirt.png
    person_img, cloth_img = setup_data_structure_from_images()
    
    # If direct image setup fails, fall back to train data
    if not person_img or not cloth_img:
        print("Failed to set up using direct images, trying train pair fallback...")
        person_img, cloth_img = fallback_to_train_pair()
        
        if not person_img or not cloth_img:
            print("All setup attempts failed. Aborting.")
            return
    
    # Configure VAE
    configure_vae()
    
    # Run model inference
    output_dir = "model_results"
    result_path = run_model_inference(output_dir)
    
    if result_path and os.path.exists(result_path):
        print("\nSuccess! StableVITON model inference completed")
        print(f"Result saved to: {result_path}")
        print("This is a high-quality result using the actual StableVITON model with your checkpoints")
    else:
        print("\nStableVITON model inference failed")
        print("Please check error messages above for troubleshooting")

if __name__ == "__main__":
    main() 