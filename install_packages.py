"""
Install all Python packages required for this project.
Run this script to set up your Python environment.
"""

import subprocess
import sys

def install_packages():
    """Install all required Python packages."""
    packages = [
        'yfinance>=0.2.28',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'requests>=2.31.0',
        'tqdm>=4.65.0',
    ]

    print("Installing Python packages for Meme Stock Detection project...")
    print("=" * 60)

    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False

    print("\n" + "=" * 60)
    print("All Python packages installed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = install_packages()
    sys.exit(0 if success else 1)