import os
import sys
import subprocess
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import shutil

def ensure_dir(path):
    """Create directory if it doesn't exist"""
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
        print(f"Created mask at {output_path}")
        return True
    except Exception as e:
        print(f"Error creating mask: {e}")
        return False

def simple_alignment(person_img_path, cloth_img_path, output_path):
    """Perform a basic alignment of clothing onto person"""
    try:
        # Open images
        person_img = Image.open(person_img_path)
        cloth_img = Image.open(cloth_img_path)
        
        # Convert to RGBA if not already
        if person_img.mode != 'RGBA':
            person_img = person_img.convert('RGBA')
        if cloth_img.mode != 'RGBA':
            cloth_img = cloth_img.convert('RGBA')
        
        # Get dimensions
        person_width, person_height = person_img.size
        cloth_width, cloth_height = cloth_img.size
        
        # Resize cloth to better fit the person
        # Resizing ratio can be adjusted for better fit
        ratio = min(person_width * 0.8 / cloth_width, person_height * 0.4 / cloth_height)
        new_cloth_width = int(cloth_width * ratio)
        new_cloth_height = int(cloth_height * ratio)
        cloth_img = cloth_img.resize((new_cloth_width, new_cloth_height), Image.Resampling.LANCZOS)
        
        # Calculate approximate upper body position
        paste_x = (person_width - new_cloth_width) // 2
        paste_y = int(person_height * 0.2)  # Position shirt about 20% down from top
        
        # Create a result image
        result_img = person_img.copy()
        
        # Use the original alpha channel of the shirt for compositing
        result_img.paste(cloth_img, (paste_x, paste_y), cloth_img)
        
        # Save result
        result_img.save(output_path)
        print(f"Basic alignment created and saved to {output_path}")
        return result_img
    except Exception as e:
        print(f"Error in simple alignment: {e}")
        return None

def prepare_test_dataset(person_img_path, cloth_img_path):
    """Prepare the required dataset structure for StableVITON inference"""
    print("Preparing dataset structure for StableVITON...")
    
    # Create necessary directories
    data_root = "StableVITON/data"
    data_test = os.path.join(data_root, "test")
    
    # Required directories for the test data
    test_dirs = [
        os.path.join(data_test, "image"),
        os.path.join(data_test, "cloth"),
        os.path.join(data_test, "cloth-mask")
    ]
    
    for dir_path in test_dirs:
        ensure_dir(dir_path)
    
    # Copy and rename person image
    person_filename = "person_01.jpg"
    cloth_filename = "cloth_01.jpg"
    
    # Destination paths
    dst_person = os.path.join(data_test, "image", person_filename)
    dst_cloth = os.path.join(data_test, "cloth", cloth_filename)
    dst_cloth_mask = os.path.join(data_test, "cloth-mask", cloth_filename)
    
    # Convert and save images in the proper format
    try:
        # Person image
        person_img = Image.open(person_img_path)
        person_img = person_img.convert('RGB')
        person_img = person_img.resize((384, 512))  # Standard size for StableVITON
        person_img.save(dst_person)
        
        # Cloth image
        cloth_img = Image.open(cloth_img_path)
        cloth_img = cloth_img.convert('RGB')
        cloth_img = cloth_img.resize((384, 512))  # Standard size
        cloth_img.save(dst_cloth)
        
        # Create and save cloth mask
        create_mask_from_image(cloth_img_path, dst_cloth_mask)
        
        # Create pairs.txt file
        pairs_path = os.path.join(data_test, "test_pairs.txt")
        with open(pairs_path, "w") as f:
            f.write(f"{person_filename} {cloth_filename}")
        
        print("Dataset preparation complete.")
        return True
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return False

def attempt_stableviton_inference():
    """Attempt to run StableVITON inference using the prepared data"""
    print("Attempting StableVITON inference...")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"proper_results_{timestamp}"
    ensure_dir(output_dir)
    
    # Check for model weights
    if not os.path.exists("VITONHD_PBE_POSE.ckpt"):
        print("Warning: Model weights not found. Full StableVITON inference not possible.")
        return None
    
    # Prepare inference command
    inference_cmd = [
        sys.executable, "StableVITON/inference.py",
        "--config_path", "StableVITON/configs/VITONHD.yaml",
        "--model_load_path", "VITONHD_PBE_POSE.ckpt",
        "--batch_size", "1",
        "--data_root_dir", "StableVITON/data",
        "--save_dir", output_dir,
        "--denoise_steps", "50",
        "--img_H", "512",
        "--img_W", "384"
    ]
    
    # Set a timeout for the process
    max_wait_time = 300  # 5 minutes max
    
    # Run inference with timeout
    try:
        print("Starting StableVITON inference process...")
        print(f"Command: {' '.join(inference_cmd)}")
        
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
                print("StableVITON inference timed out after 5 minutes.")
                return None
            time.sleep(1)
        
        # Check result
        if process.returncode != 0:
            stderr = process.stderr.read()
            print(f"StableVITON inference failed with error:\n{stderr}")
            return None
        else:
            print(f"StableVITON inference completed successfully.")
            # Check for results
            result_files = []
            if os.path.exists(output_dir):
                result_files = [f for f in os.listdir(output_dir) 
                               if f.endswith('.jpg') or f.endswith('.png')]
            
            if result_files:
                result_path = os.path.join(output_dir, result_files[0])
                print(f"Found StableVITON result: {result_path}")
                
                # Copy to a standard name
                final_path = "stableviton_result.png"
                shutil.copy(result_path, final_path)
                print(f"Copied StableVITON result to {final_path}")
                
                return final_path
            else:
                print("No result files found in StableVITON output directory.")
                return None
    except Exception as e:
        print(f"Error attempting StableVITON inference: {e}")
        return None

def create_final_visualization(person_img_path, cloth_img_path, basic_result_path, stableviton_result_path=None):
    """Create a final comparison visualization with all images"""
    print("Creating final comparison visualization...")
    
    # Output path
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = f"final_tryon_comparison_{timestamp}.png"
    
    # Load images
    person_img = Image.open(person_img_path).convert('RGB')
    cloth_img = Image.open(cloth_img_path).convert('RGB')
    basic_result = Image.open(basic_result_path).convert('RGB')
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Original person image
    plt.subplot(2, 2, 1)
    plt.imshow(np.array(person_img))
    plt.title("Person Image")
    plt.axis('off')
    
    # Original clothing image
    plt.subplot(2, 2, 2)
    plt.imshow(np.array(cloth_img))
    plt.title("Clothing Item")
    plt.axis('off')
    
    # Basic alignment result
    plt.subplot(2, 2, 3)
    plt.imshow(np.array(basic_result))
    plt.title("Basic Image Alignment")
    plt.axis('off')
    
    # StableVITON result (if available)
    plt.subplot(2, 2, 4)
    if stableviton_result_path and os.path.exists(stableviton_result_path):
        try:
            stableviton_result = Image.open(stableviton_result_path).convert('RGB')
            plt.imshow(np.array(stableviton_result))
            plt.title("StableVITON Result")
        except Exception as e:
            print(f"Error loading StableVITON result: {e}")
            plt.text(0.5, 0.5, "StableVITON Result\n(Failed to load)", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, "StableVITON Result\n(Not available)", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, transform=plt.gca().transAxes)
    plt.axis('off')
    
    # Add overall title
    plt.suptitle("Virtual Try-On Comparison", fontsize=16)
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_path)
    plt.close()
    
    print(f"Final comparison visualization saved to {output_path}")
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate virtual try-on results using multiple methods")
    parser.add_argument("--person", type=str, default="zz.png", help="Path to person image")
    parser.add_argument("--garment", type=str, default="shirt.png", help="Path to garment image")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting complete virtual try-on process...")
    print(f"Person image: {args.person}")
    print(f"Garment image: {args.garment}")
    
    # Create results directory
    ensure_dir("results")
    
    # Step 1: Generate basic alignment
    basic_result_path = "results/basic_alignment_result.png"
    basic_result = simple_alignment(args.person, args.garment, basic_result_path)
    
    # Step 2: Attempt StableVITON inference
    # First prepare the dataset
    prepare_test_dataset(args.person, args.garment)
    
    # Configure VAE if available
    if os.path.exists("VITONHD_VAE_finetuning.ckpt"):
        print("Configuring VAE fine-tuning...")
        try:
            vae_cmd = [
                sys.executable, "StableVITON/use_vae.py",
                "--config_file", "StableVITON/configs/VITONHD.yaml",
                "--vae_ckpt", "VITONHD_VAE_finetuning.ckpt"
            ]
            subprocess.run(vae_cmd, check=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to configure VAE. Continuing with default settings.")
    
    # Try running StableVITON
    stableviton_result_path = attempt_stableviton_inference()
    
    # Step 3: Create final visualization comparing all results
    final_viz_path = create_final_visualization(
        args.person, 
        args.garment, 
        basic_result_path, 
        stableviton_result_path
    )
    
    print("\nProcess completed!")
    print(f"Final comparison visualization: {final_viz_path}")
    print(f"Basic alignment result: {basic_result_path}")
    
    if stableviton_result_path:
        print(f"StableVITON result: {stableviton_result_path}")
    else:
        print("StableVITON inference was not successful.")
    
    print("\nYou can use any of these images for your presentation.")

if __name__ == "__main__":
    main() 