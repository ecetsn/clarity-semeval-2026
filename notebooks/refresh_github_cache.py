#!/usr/bin/env python3
"""
GitHub Cache Refresh Script for Colab
This script removes the old repository and clones fresh from GitHub
to ensure you have the latest code.
"""

import shutil
import os
import subprocess
import sys
from pathlib import Path

# Repository configuration
repo_dir = Path('/content/semeval-context-tree-modular')
repo_url = 'https://github.com/EonTechie/semeval-context-tree-modular.git'

print("="*80)
print("GITHUB CACHE REFRESH")
print("="*80)

# Step 1: Remove old repository if exists
if repo_dir.exists():
    print(f"\n1. Removing old repository: {repo_dir}")
    try:
        shutil.rmtree(repo_dir)
        print("   ✓ Old repository removed")
    except Exception as e:
        print(f"   ⚠ Error removing old repository: {e}")
        print("   Trying to remove with force...")
        subprocess.run(['rm', '-rf', str(repo_dir)], check=False)
else:
    print(f"\n1. No existing repository found at {repo_dir}")

# Step 2: Clone fresh repository
print(f"\n2. Cloning fresh repository from GitHub...")
print(f"   URL: {repo_url}")
print(f"   Destination: {repo_dir}")

max_retries = 3
clone_success = False

for attempt in range(max_retries):
    try:
        print(f"\n   Attempt {attempt + 1}/{max_retries}...")
        result = subprocess.run(
            ['git', 'clone', repo_url, str(repo_dir)],
            cwd='/content',
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("   ✓ Repository cloned successfully")
            clone_success = True
            break
        else:
            print(f"   ⚠ Git clone failed: {result.stderr}")
            if attempt < max_retries - 1:
                print("   Retrying in 3 seconds...")
                import time
                time.sleep(3)
    except subprocess.TimeoutExpired:
        print(f"   ⚠ Clone timeout (attempt {attempt + 1})")
        if attempt < max_retries - 1:
            print("   Retrying...")
    except Exception as e:
        print(f"   ⚠ Error during clone: {e}")
        if attempt < max_retries - 1:
            print("   Retrying...")

if not clone_success:
    print("\n   ⚠ Git clone failed. Trying ZIP download as fallback...")
    try:
        import requests
        import zipfile
        
        zip_url = 'https://github.com/EonTechie/semeval-context-tree-modular/archive/refs/heads/main.zip'
        zip_path = '/tmp/repo.zip'
        
        print("   Downloading ZIP archive...")
        response = requests.get(zip_url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("   Extracting ZIP archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('/content')
        
        extracted_dir = Path('/content/semeval-context-tree-modular-main')
        if extracted_dir.exists():
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            extracted_dir.rename(repo_dir)
        
        os.remove(zip_path)
        print("   ✓ Repository downloaded and extracted successfully")
        clone_success = True
    except Exception as e:
        print(f"   ✗ ZIP download also failed: {e}")
        raise RuntimeError("Failed to obtain repository from GitHub")

# Step 3: Verify repository structure
print(f"\n3. Verifying repository structure...")
if not repo_dir.exists():
    raise RuntimeError(f"Repository directory not found: {repo_dir}")

required_paths = [
    repo_dir / 'src',
    repo_dir / 'src' / 'storage' / 'manager.py',
    repo_dir / 'src' / 'features' / 'extraction.py',
    repo_dir / 'src' / 'utils' / '__init__.py',
    repo_dir / 'src' / 'utils' / 'reproducibility.py',
]

all_ok = True
for path in required_paths:
    if path.exists():
        print(f"   ✓ {path.relative_to(repo_dir)}")
    else:
        print(f"   ✗ {path.relative_to(repo_dir)} - MISSING!")
        all_ok = False

if not all_ok:
    raise RuntimeError("Repository structure incomplete. Some required files are missing.")

# Step 4: Update Python path
print(f"\n4. Updating Python path...")
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))
    print(f"   ✓ Added {repo_dir} to Python path")
else:
    print(f"   ✓ {repo_dir} already in Python path")

# Step 5: Verify imports
print(f"\n5. Verifying imports...")
try:
    from src.storage.manager import StorageManager
    print("   ✓ StorageManager imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import StorageManager: {e}")
    raise

try:
    from src.utils.reproducibility import set_all_seeds
    print("   ✓ set_all_seeds imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import set_all_seeds: {e}")
    raise

try:
    from src.features.extraction import (
        featurize_hf_dataset_in_batches_v2,
        featurize_model_independent_features
    )
    print("   ✓ Feature extraction functions imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import feature extraction functions: {e}")
    raise

print(f"\n{'='*80}")
print("✓ GITHUB CACHE REFRESH COMPLETE")
print(f"{'='*80}")
print(f"\nRepository location: {repo_dir}")
print(f"Python path updated: {str(repo_dir) in sys.path}")
print("\nYou can now use:")
print("  from src.storage.manager import StorageManager")
print("  from src.utils.reproducibility import set_all_seeds")
print("  from src.features.extraction import featurize_hf_dataset_in_batches_v2")

