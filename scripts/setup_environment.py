import os
import sys
import subprocess
from pathlib import Path

VENV_DIR = ".venv"

def check_venv():
    try:
        import venv
        return True
    except ImportError:
        print("ERROR: Python's venv module is not available.")
        print("Please install the appropriate package for your system.")
        return False

def create_venv():
    if not Path(VENV_DIR).exists():
        print("Creating virtual environment...")
        subprocess.check_call(
            [sys.executable, "-m", "venv", VENV_DIR]
        )
    else:
        print("Virtual environment already exists.")

def install_requirements():
    if os.name == "nt":
        pip_path = os.path.join(
            VENV_DIR,
            "Scripts",
            "pip.exe"
        )
    else:
        pip_path = os.path.join(
            VENV_DIR,
            "bin",
            "pip"
        )

    print("Installing dependencies...")
    subprocess.check_call(
        [pip_path, "install", "-r", "requirements.txt"]
    )

def main():
    if not check_venv():
        sys.exit(1)

    create_venv()
    install_requirements()

    print("\nSetup completed successfully.")

if __name__ == "__main__":
    main()