import os
import sys
import argparse
import subprocess
import importlib.util

def check_module(module_name):
    """Check if a Python module is installed"""
    return importlib.util.find_spec(module_name) is not None

def install_dependencies():
    """Install dependencies using the installation script"""
    print("Checking and installing dependencies...")
    # Run the dependency installation script
    if os.path.exists("install_dependencies_tryon.py"):
        try:
            # Import and run the script
            subprocess.run([sys.executable, "install_dependencies_tryon.py"], check=True)
            return True
        except subprocess.SubprocessError as e:
            print(f"Error running dependency installation: {str(e)}")
            return False
    else:
        print("Dependency installation script not found.")
        return False

def run_tryon(person_img, cloth_img):
    """Run the virtual try-on process"""
    print("\nRunning virtual try-on...")
    
    # Check if the improved_tryon.py script exists
    if not os.path.exists("improved_tryon.py"):
        print("Error: improved_tryon.py script not found.")
        return False
    
    # Run the try-on script
    try:
        cmd = [sys.executable, "improved_tryon.py", "--person", person_img, "--cloth", cloth_img]
        subprocess.run(cmd, check=True)
        print("\nVirtual try-on completed successfully!")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error running virtual try-on: {str(e)}")
        return False

def check_required_files(person_img, cloth_img):
    """Check if the required input files exist"""
    if not os.path.exists(person_img):
        print(f"Error: Person image '{person_img}' not found.")
        return False
    
    if not os.path.exists(cloth_img):
        print(f"Error: Clothing image '{cloth_img}' not found.")
        return False
    
    return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the Virtual Try-On process")
    parser.add_argument("--person", type=str, default="zz.png", 
                      help="Path to person image (default: zz.png)")
    parser.add_argument("--cloth", type=str, default="shirt.png", 
                      help="Path to clothing image (default: shirt.png)")
    return parser.parse_args()

def main():
    """Main function to run the installation and try-on process"""
    print("=" * 60)
    print("        VIRTUAL TRY-ON SYSTEM - LOUIS VUITTON DEMO")
    print("=" * 60)
    
    # Parse arguments
    args = parse_args()
    
    # Check if input files exist
    if not check_required_files(args.person, args.cloth):
        return
    
    # Install dependencies if needed
    install_dependencies()
    
    # Run the try-on process
    success = run_tryon(args.person, args.cloth)
    
    if success:
        print("\nResults are saved in the 'results' directory.")
        print("Check the visualization images to see the try-on comparison.")
    else:
        print("\nVirtual try-on process encountered issues.")
        print("Check the error messages above for more information.")
    
    print("\nThank you for using the Virtual Try-On System!")

if __name__ == "__main__":
    main() 