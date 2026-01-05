"""
Reproducibility utilities for setting random seeds across all libraries
"""
import random
import numpy as np
import torch
import os


def set_all_seeds(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - HuggingFace Transformers (if available)
    
    Args:
        seed: Random seed value (default: 42)
        deterministic: If True, enable deterministic mode for PyTorch (slower but fully reproducible)
    
    Note:
        - For full reproducibility, set deterministic=True
        - Deterministic mode may be slower and may not be available on all GPUs
        - Some operations (e.g., certain CUDA operations) may still be non-deterministic
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # PyTorch deterministic mode (slower but fully reproducible)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: Some operations may still be non-deterministic on certain GPUs
        # If you encounter issues, set deterministic=False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # HuggingFace Transformers (if available)
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass  # transformers not available
    
    print(f"✓ Reproducibility seeds set to {seed}")
    if deterministic:
        print("✓ PyTorch deterministic mode enabled (may be slower)")
    else:
        print(" PyTorch deterministic mode disabled (faster but less reproducible)")

