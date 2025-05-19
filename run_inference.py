import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Run StableVITON inference')
    parser.add_argument('--data_dir', type=str, default='test', help='Directory with test data')
    parser.add_argument('--output_dir', type=str, default='results_inference', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--pairs_file', type=str, default='test_pairs.txt', help='Text file with test pairs')
    parser.add_argument('--use_vae', action='store_true', help='Use VAE fine-tuning checkpoint')
    return parser.parse_args()

def prepare_data_structure(args):
    """Creates symbolic links to organize data for StableVITON inference."""
    print("Preparing data structure...")
    
    # Create necessary directories
    os.makedirs("StableVITON/data/test/image", exist_ok=True)
    os.makedirs("StableVITON/data/test/cloth", exist_ok=True)
    os.makedirs("StableVITON/data/test/cloth-mask", exist_ok=True)
    os.makedirs("StableVITON/data/test/image-densepose", exist_ok=True)
    os.makedirs("StableVITON/data/test/agnostic-v3.2", exist_ok=True)
    os.makedirs("StableVITON/data/test/agnostic-mask", exist_ok=True)
    
    # Copy or symlink data files based on pairs file
    with open(args.pairs_file, 'r') as f:
        pairs = f.readlines()
    
    for i, pair in enumerate(pairs[:min(10, len(pairs))]):  # Process first 10 pairs for testing
        try:
            pair = pair.strip()
            if not pair:
                continue
                
            parts = pair.split()
            if len(parts) < 2:
                print(f"Skipping invalid pair: {pair}")
                continue
                
            image_name, cloth_name = parts[0], parts[1]
            
            # Copy necessary files (using PowerShell commands for Windows)
            # Person image
            src_path = os.path.join(args.data_dir, "image", image_name)
            dst_path = os.path.join("StableVITON", "data", "test", "image", image_name)
            if os.path.exists(src_path):
                copy_cmd = f'Copy-Item -Path "{src_path}" -Destination "{dst_path}" -Force'
                subprocess.run(["powershell", "-Command", copy_cmd], check=False)
            
            # Cloth image
            src_path = os.path.join(args.data_dir, "cloth", cloth_name)
            dst_path = os.path.join("StableVITON", "data", "test", "cloth", cloth_name)
            if os.path.exists(src_path):
                copy_cmd = f'Copy-Item -Path "{src_path}" -Destination "{dst_path}" -Force'
                subprocess.run(["powershell", "-Command", copy_cmd], check=False)
            
            # Cloth mask
            src_path = os.path.join(args.data_dir, "cloth-mask", cloth_name)
            dst_path = os.path.join("StableVITON", "data", "test", "cloth-mask", cloth_name)
            if os.path.exists(src_path):
                copy_cmd = f'Copy-Item -Path "{src_path}" -Destination "{dst_path}" -Force'
                subprocess.run(["powershell", "-Command", copy_cmd], check=False)
            
            # Densepose image
            src_path = os.path.join(args.data_dir, "image-densepose", image_name)
            dst_path = os.path.join("StableVITON", "data", "test", "image-densepose", image_name)
            if os.path.exists(src_path):
                copy_cmd = f'Copy-Item -Path "{src_path}" -Destination "{dst_path}" -Force'
                subprocess.run(["powershell", "-Command", copy_cmd], check=False)
            
            # Agnostic image
            src_path = os.path.join(args.data_dir, "agnostic-v3.2", image_name)
            dst_path = os.path.join("StableVITON", "data", "test", "agnostic-v3.2", image_name)
            if os.path.exists(src_path):
                copy_cmd = f'Copy-Item -Path "{src_path}" -Destination "{dst_path}" -Force'
                subprocess.run(["powershell", "-Command", copy_cmd], check=False)
            
            # Agnostic mask
            src_path = os.path.join(args.data_dir, "agnostic-mask", image_name)
            dst_path = os.path.join("StableVITON", "data", "test", "agnostic-mask", image_name)
            if os.path.exists(src_path):
                copy_cmd = f'Copy-Item -Path "{src_path}" -Destination "{dst_path}" -Force'
                subprocess.run(["powershell", "-Command", copy_cmd], check=False)
                
            print(f"Processed pair {i+1}/{min(10, len(pairs))}: {image_name} + {cloth_name}")
            
        except Exception as e:
            print(f"Error processing pair {pair}: {e}")
    
    print("Data preparation completed.")

def configure_vae(use_vae):
    """Configure the model to use VAE fine-tuning checkpoint if specified."""
    if use_vae:
        print("Configuring to use VAE fine-tuning checkpoint...")
        vae_cmd = [
            "python", "StableVITON/use_vae.py",
            "--config_file", "StableVITON/configs/VITONHD.yaml",
            "--vae_ckpt", "StableVITON/ckpts/VITONHD_VAE_finetuning.ckpt"
        ]
        
        process = subprocess.Popen(vae_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        
        if process.returncode != 0:
            print(f"VAE configuration failed with return code {process.returncode}")
            return False
        else:
            print("VAE configuration completed successfully")
            return True
    return True

def run_inference(args):
    """Run the StableVITON inference script."""
    print("Running inference with StableVITON...")
    
    # Install required packages if needed
    try:
        print("Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml", "omegaconf", "einops"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing required packages: {e}")
        return False
    
    # Setup command with proper paths and arguments
    inference_cmd = [
        "python", "StableVITON/inference.py",
        "--config_path", "StableVITON/configs/VITONHD.yaml",
        "--model_load_path", "StableVITON/ckpts/VITONHD_PBE_pose.ckpt",
        "--batch_size", str(args.batch_size),
        "--data_root_dir", "StableVITON/data",
        "--save_dir", args.output_dir,
        "--denoise_steps", "50",
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
        return False
    else:
        print(f"Inference completed successfully. Results saved to {args.output_dir}")
        return True

def main():
    args = parse_args()
    prepare_data_structure(args)
    
    if args.use_vae:
        if not configure_vae(args.use_vae):
            print("Failed to configure VAE. Continuing with default settings.")
    
    run_inference(args)

if __name__ == "__main__":
    main() 