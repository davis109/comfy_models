import subprocess
import sys
import os

def run_command(command):
    """Run a command and return its output"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_and_install_package(package_name, version=None):
    """Check if a package is installed and install it if it's not"""
    package_spec = f"{package_name}{f'=={version}' if version else ''}"
    
    print(f"Checking for {package_spec}...")
    
    # Try to import the package
    try:
        __import__(package_name.replace('-', '_'))
        print(f"✓ {package_name} is already installed")
        return True
    except ImportError:
        print(f"× {package_name} is not installed. Installing...")
        success, output = run_command(f"{sys.executable} -m pip install {package_spec}")
        if success:
            print(f"✓ Successfully installed {package_spec}")
            return True
        else:
            print(f"× Failed to install {package_spec}: {output}")
            return False

def main():
    """Install all required dependencies"""
    print("Installing dependencies for Virtual Try-On...")
    
    dependencies = [
        ("numpy", None),  # Latest version
        ("Pillow", None),  # For image processing
        ("matplotlib", None),  # For visualization
        ("opencv-python", None)  # For image segmentation
    ]
    
    # Ensure pip is up to date
    print("Updating pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install each dependency
    all_success = True
    for package, version in dependencies:
        success = check_and_install_package(package, version)
        all_success = all_success and success
    
    if all_success:
        print("\n✅ All dependencies installed successfully!")
        print("\nYou can now run the Virtual Try-On application with:")
        print(f"  {sys.executable} improved_tryon.py --person [person_image] --cloth [cloth_image]")
    else:
        print("\n❌ Some dependencies could not be installed. Please check the error messages above.")

if __name__ == "__main__":
    main() 