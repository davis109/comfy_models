import subprocess
import sys

def install_deps():
    """Install required dependencies for StableVITON"""
    print("Installing required dependencies...")
    
    # List of required packages
    dependencies = [
        "omegaconf",
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "opencv-python",
        "matplotlib",
        "einops",
        "pytorch-lightning",
    ]
    
    for package in dependencies:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
    
    print("Dependency installation complete!")

if __name__ == "__main__":
    install_deps() 