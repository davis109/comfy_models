import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tqdm

class StableVITONDataset(Dataset):
    def __init__(self, data_root_dir, pairs_file, img_size=(512, 384), is_test=False):
        """Dataset for StableVITON virtual try-on.
        
        Args:
            data_root_dir: Path to the data directory (train or test)
            pairs_file: Path to the pairs file (train_pairs.txt or test_pairs.txt)
            img_size: Tuple of (height, width) for resizing images
            is_test: Whether this is a test dataset
        """
        self.data_root = data_root_dir
        self.img_size = img_size
        self.is_test = is_test
        
        # Read pairs file
        self.im_names = []
        self.c_names = []
        try:
            with open(pairs_file, "r") as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    self.im_names.append(im_name)
                    self.c_names.append(c_name)
            print(f"Successfully loaded {len(self.im_names)} pairs from {pairs_file}")
        except FileNotFoundError:
            print(f"Warning: Pairs file {pairs_file} not found!")
        except Exception as e:
            print(f"Error reading pairs file: {e}")
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[idx]
        
        # Load image
        image_path = os.path.join(self.data_root, "image", img_fn)
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image_tensor = torch.zeros((3, *self.img_size))
        
        # Load cloth
        cloth_path = os.path.join(self.data_root, "cloth", cloth_fn)
        try:
            cloth = Image.open(cloth_path).convert('RGB')
            cloth_tensor = self.transform(cloth)
        except Exception as e:
            print(f"Error loading cloth {cloth_path}: {e}")
            cloth_tensor = torch.zeros((3, *self.img_size))
        
        # Load cloth mask
        cloth_mask_path = os.path.join(self.data_root, "cloth-mask", cloth_fn)
        try:
            if os.path.exists(cloth_mask_path):
                cloth_mask = Image.open(cloth_mask_path).convert('L')
                cloth_mask_tensor = self.mask_transform(cloth_mask)
            else:
                cloth_mask_tensor = torch.ones((1, *self.img_size))
        except Exception as e:
            print(f"Error loading cloth mask {cloth_mask_path}: {e}")
            cloth_mask_tensor = torch.ones((1, *self.img_size))
        
        # Load agnostic
        agnostic_path = os.path.join(self.data_root, "agnostic-v3.2", img_fn)
        try:
            if os.path.exists(agnostic_path):
                agnostic = Image.open(agnostic_path).convert('RGB')
                agnostic_tensor = self.transform(agnostic)
            else:
                agnostic_tensor = torch.zeros_like(image_tensor)
        except Exception as e:
            print(f"Error loading agnostic {agnostic_path}: {e}")
            agnostic_tensor = torch.zeros_like(image_tensor)
        
        # Load densepose
        densepose_path = os.path.join(self.data_root, "image-densepose", img_fn)
        try:
            if os.path.exists(densepose_path):
                densepose = Image.open(densepose_path).convert('RGB')
                densepose_tensor = self.transform(densepose)
            else:
                densepose_tensor = torch.zeros_like(image_tensor)
        except Exception as e:
            print(f"Error loading densepose {densepose_path}: {e}")
            densepose_tensor = torch.zeros_like(image_tensor)
        
        return {
            'image': image_tensor,
            'cloth': cloth_tensor,
            'cloth_mask': cloth_mask_tensor,
            'agnostic': agnostic_tensor,
            'densepose': densepose_tensor,
            'image_name': img_fn,
            'cloth_name': cloth_fn
        }

def show_tensor_image(tensor, title=None):
    """Display a tensor as an image."""
    img = tensor.clone()
    if len(img.shape) == 4:
        img = img[0]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)  # Unnormalize
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

def parse_args():
    parser = argparse.ArgumentParser(description='StableVITON Demo')
    parser.add_argument('--data_root', type=str, default='test', help='Path to data directory (train or test)')
    parser.add_argument('--pairs_file', type=str, default='test_pairs.txt', help='Path to pairs file')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--is_test', action='store_true', help='Whether to use test dataset')
    parser.add_argument('--skip_missing', action='store_true', help='Skip samples with missing files')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_root):
        print(f"Error: Data directory {args.data_root} not found!")
        print("Please make sure the data directory exists and contains the required folders.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = StableVITONDataset(
        data_root_dir=args.data_root,
        pairs_file=args.pairs_file,
        is_test=args.is_test
    )
    
    if len(dataset) == 0:
        print("Error: No valid samples found in the dataset.")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Visualizing {min(args.num_samples, len(dataset))} samples...")
    
    # Visualize samples
    samples_processed = 0
    errors = 0
    
    progress_bar = tqdm.tqdm(dataloader, total=min(args.num_samples, len(dataset)))
    for i, batch in enumerate(progress_bar):
        if samples_processed >= args.num_samples:
            break
            
        progress_bar.set_description(f"Processing sample {i+1}")
        
        try:
            image = batch['image']
            cloth = batch['cloth']
            cloth_mask = batch['cloth_mask']
            agnostic = batch['agnostic']
            densepose = batch['densepose']
            image_name = batch['image_name'][0]
            cloth_name = batch['cloth_name'][0]
            
            # Check if any tensors are all zeros (missing files)
            if args.skip_missing and (
                torch.all(image == 0) or 
                torch.all(cloth == 0) or 
                torch.all(agnostic == 0) or 
                torch.all(densepose == 0)
            ):
                print(f"Skipping sample {i+1} due to missing files")
                continue
            
            # Create a visualization
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            show_tensor_image(image, "Person Image")
            
            plt.subplot(2, 3, 2)
            show_tensor_image(cloth, "Cloth Image")
            
            plt.subplot(2, 3, 3)
            plt.imshow(cloth_mask[0, 0].cpu().numpy(), cmap='gray')
            plt.title("Cloth Mask")
            plt.axis('off')
            
            plt.subplot(2, 3, 4)
            show_tensor_image(agnostic, "Agnostic")
            
            plt.subplot(2, 3, 5)
            show_tensor_image(densepose, "DensePose")
            
            # This would be where the generated image goes in a full implementation
            plt.subplot(2, 3, 6)
            plt.text(0.5, 0.5, "Virtual Try-On\n(Simulated Result)", 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12, transform=plt.gca().transAxes)
            plt.axis('off')
            
            plt.suptitle(f"Sample {samples_processed+1}: {image_name} with {cloth_name}")
            plt.tight_layout()
            
            # Save the visualization
            output_filename = f"sample_{samples_processed+1}_{image_name}_{cloth_name}.png"
            output_path = os.path.join(args.output_dir, output_filename)
            plt.savefig(output_path)
            plt.close()
            
            samples_processed += 1
            progress_bar.set_postfix({"saved": output_filename})
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            errors += 1
            plt.close()
    
    print(f"\nProcessing complete: {samples_processed} samples saved to {args.output_dir}")
    if errors > 0:
        print(f"Encountered {errors} errors during processing")

if __name__ == "__main__":
    main() 