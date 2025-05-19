import os
import subprocess
import sys

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    # Create output directory
    output_dir = "results_inference"
    ensure_dir(output_dir)
    
    # Install required packages
    print("Installing required packages...")
    packages = ["pyyaml", "omegaconf", "einops", "opencv-python"]
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package}")
    
    # Make sure the checkpoint directory exists
    ensure_dir("StableVITON/ckpts")
    
    # Make sure the data directory exists
    data_dirs = [
        "StableVITON/data/test/image",
        "StableVITON/data/test/cloth",
        "StableVITON/data/test/cloth_mask",
        "StableVITON/data/test/image-densepose",
        "StableVITON/data/test/agnostic",
        "StableVITON/data/test/agnostic-mask"
    ]
    for dir_path in data_dirs:
        ensure_dir(dir_path)
    
    # Copy test images and garments (just a few for testing)
    test_mappings = [
        ("image", "image"),
        ("cloth", "cloth"),
        ("cloth-mask", "cloth_mask"),
        ("image-densepose", "image-densepose"),
        ("agnostic-v3.2", "agnostic"),
        ("agnostic-mask", "agnostic-mask")
    ]
    
    for src_dir_name, dst_dir_name in test_mappings:
        # Copy just a few items from each directory
        src_dir = os.path.join("test", src_dir_name)
        dst_dir = os.path.join("StableVITON", "data", "test", dst_dir_name)
        
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} not found")
            continue
            
        files = os.listdir(src_dir)[:3]  # Just copy first 3 files
        for file in files:
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, file)
            copy_cmd = f'Copy-Item -Path "{src_path}" -Destination "{dst_path}" -Force'
            try:
                subprocess.run(["powershell", "-Command", copy_cmd], check=False)
                print(f"Copied {src_path} to {dst_path}")
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
    
    # Configure to use VAE fine-tuning
    print("Configuring to use VAE fine-tuning...")
    try:
        vae_cmd = [
            "python", "StableVITON/use_vae.py",
            "--config_file", "StableVITON/configs/VITONHD.yaml",
            "--vae_ckpt", "StableVITON/ckpts/VITONHD_VAE_finetuning.ckpt"
        ]
        subprocess.run(vae_cmd, check=True)
    except subprocess.CalledProcessError:
        print("Warning: Failed to configure VAE. Continuing with default settings.")
    
    # Run inference
    print("Running inference with StableVITON...")
    inference_cmd = [
        "python", "StableVITON/inference.py",
        "--config_path", "StableVITON/configs/VITONHD.yaml",
        "--model_load_path", "StableVITON/ckpts/VITONHD_PBE_pose.ckpt",
        "--batch_size", "1",  # Small batch size to avoid memory issues
        "--data_root_dir", "StableVITON/data",
        "--save_dir", output_dir,
        "--denoise_steps", "20",  # Fewer steps for faster results
        "--img_H", "512",
        "--img_W", "384"
    ]
    
    # Run the command
    process = subprocess.Popen(inference_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    
    if process.returncode != 0:
        print(f"Inference failed with return code {process.returncode}")
    else:
        print(f"Inference completed successfully. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 