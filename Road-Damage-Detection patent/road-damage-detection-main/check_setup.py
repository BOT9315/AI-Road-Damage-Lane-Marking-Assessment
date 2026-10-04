"""
check_setup.py
===============
Run this after `pip install -r requirements.txt` to verify the
environment is correctly configured before training or running the app.

Usage:
    python check_setup.py
"""

import importlib
import sys

REQUIRED = [
    "ultralytics", "torch", "cv2", "numpy", "yaml",
    "pandas", "sklearn", "matplotlib", "streamlit", "tqdm",
]

def main():
    print("Checking required packages...\n")
    missing = []
    for pkg in REQUIRED:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {pkg:<12} {version}")
        except ImportError:
            print(f"  [MISSING] {pkg}")
            missing.append(pkg)

    print()
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    if missing:
        print(f"\n{len(missing)} package(s) missing. Run: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\nAll dependencies satisfied. You're ready to train / run inference.")


if __name__ == "__main__":
    main()