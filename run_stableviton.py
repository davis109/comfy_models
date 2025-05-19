import os
import sys
import argparse
import subprocess
import shutil
from PIL import Image
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from stableviton_dataset_prep import prepare_full_dataset

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def run_stableviton_inference():
    """Run StableVITON inference using the prepared data"""
    print("Running StableVITON inference...")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"stableviton_results_{timestamp}"
    ensure_dir(output_dir)
    
    # Check for model weights
    model_path = "StableVITON/ckpts/VITONHD_PBE_pose.ckpt"
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return None
    
    # Prepare inference command
    python_exe = sys.executable
    inference_script = "StableVITON/inference.py"
    config_path = "StableVITON/configs/VITONHD.yaml"
    
    inference_cmd = [
        python_exe, inference_script,
        "--config_path", config_path,
        "--model_load_path", model_path,
        "--batch_size", "1",
        "--data_root_dir", "StableVITON/data",
        "--save_dir", output_dir,
        "--denoise_steps", "50",
        "--img_H", "512",
        "--img_W", "384"
    ]
    
    # Run inference
    try:
        print("Starting StableVITON inference...")
        print(f"Command: {' '.join(inference_cmd)}")
        
        process = subprocess.Popen(
            inference_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stream the output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get the return code
        return_code = process.poll()
        
        # Check for errors
        if return_code != 0:
            stderr = process.stderr.read()
            print(f"StableVITON inference failed with error:\n{stderr}")
            return None
        
        # Look for the result file
        result_dir = os.path.join(output_dir, "pair")
        if os.path.exists(result_dir):
            result_files = [f for f in os.listdir(result_dir) 
                           if f.endswith('.jpg') or f.endswith('.png')]
            
            if result_files:
                result_path = os.path.join(result_dir, result_files[0])
                print(f"Found StableVITON result: {result_path}")
                
                # Copy to a standard name in results folder
                ensure_dir("results")
                final_path = f"results/stableviton_result_{timestamp}.jpg"
                shutil.copy(result_path, final_path)
                print(f"Copied StableVITON result to {final_path}")
                
                return final_path
            else:
                print("No result files found in StableVITON output directory.")
                return None
        else:
            print(f"Result directory {result_dir} not found.")
            return None
    except Exception as e:
        print(f"Error running StableVITON inference: {e}")
        return None

def create_visualization(person_img_path, cloth_img_path, result_img_path):
    """Create a visualization of the input and output images"""
    print("Creating visualization...")
    
    # Create output directory
    ensure_dir("results")
    
    # Output path with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = f"results/stableviton_visualization_{timestamp}.png"
    
    # Load images
    person_img = Image.open(person_img_path)
    cloth_img = Image.open(cloth_img_path)
    result_img = Image.open(result_img_path)
    
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
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "PIL", "numpy", "matplotlib", "cv2", "torch", "torchvision", 
        "omegaconf", "tqdm", "einops", "transformers", "kornia", 
        "skimage", "pytorch_lightning", "diffusers"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required dependencies:", ", ".join(missing_packages))
        print("\nPlease install dependencies first by running:")
        print("    powershell -ExecutionPolicy Bypass -File install_stableviton_deps.ps1")
        return False
    
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="StableVITON Virtual Try-On")
    parser.add_argument("--person", type=str, default="zz.png", help="Path to person image")
    parser.add_argument("--cloth", type=str, default="shirt.png", help="Path to clothing image")
    return parser.parse_args()

def main():
    """Main function to run StableVITON inference"""
    print("=" * 60)
    print("           STABLEVITON VIRTUAL TRY-ON SYSTEM")
    print("=" * 60)
    
    # Parse arguments
    args = parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.person):
        print(f"Error: Person image '{args.person}' not found.")
        return
    
    if not os.path.exists(args.cloth):
        print(f"Error: Clothing image '{args.cloth}' not found.")
        return
    
    print(f"Processing person image: {args.person}")
    print(f"Processing clothing image: {args.cloth}")
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    # Step 1: Prepare full dataset with enhanced preparation
    if not prepare_full_dataset(args.person, args.cloth):
        print("Failed to prepare dataset. Exiting.")
        return
    
    # Step 2: Run StableVITON inference
    result_path = run_stableviton_inference()
    if not result_path:
        print("StableVITON inference failed. Exiting.")
        return
    
    # Step 3: Create visualization
    viz_path = create_visualization(args.person, args.cloth, result_path)
    
    print("\nStableVITON virtual try-on completed successfully!")
    print(f"Result saved to: {result_path}")
    print(f"Visualization saved to: {viz_path}")
    print("\nThank you for using StableVITON!")

if __name__ == "__main__":
    main() 