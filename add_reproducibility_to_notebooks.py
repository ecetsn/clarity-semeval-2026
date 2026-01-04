"""
Script to add reproducibility setup cell to all notebooks
"""
import json
from pathlib import Path

REPRODUCIBILITY_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# REPRODUCIBILITY SETUP: Set Random Seeds for All Libraries\n",
        "# ============================================================================\n",
        "# This cell sets random seeds for Python, NumPy, PyTorch, and HuggingFace\n",
        "# to ensure reproducible results across all runs.\n",
        "# \n",
        "# IMPORTANT: Run this cell FIRST before any other code that uses randomness.\n",
        "# Seed value: 42 (same as used in all other parts of the pipeline)\n",
        "\n",
        "from src.utils.reproducibility import set_all_seeds\n",
        "\n",
        "# Set all random seeds to 42 for full reproducibility\n",
        "# deterministic=True ensures PyTorch operations are deterministic (slower but fully reproducible)\n",
        "set_all_seeds(seed=42, deterministic=True)\n",
        "\n",
        "print(\"✓ Reproducibility configured: All random seeds set to 42\")\n",
        "print(\"✓ PyTorch deterministic mode enabled\")\n",
        "print(\"\\nNOTE: If you encounter performance issues or non-deterministic behavior,\")\n",
        "print(\"      you can set deterministic=False in set_all_seeds() call above.\")\n"
    ]
}

def add_reproducibility_cell(notebook_path: Path):
    """Add reproducibility cell after the first setup cell"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    
    # Find the first setup cell (contains "SETUP" in source)
    setup_idx = None
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'SETUP' in source.upper() and ('Repository Clone' in source or 'setup' in source.lower()):
                setup_idx = i
                break
    
    if setup_idx is None:
        print(f"⚠ Warning: Could not find setup cell in {notebook_path.name}")
        return False
    
    # Check if reproducibility cell already exists
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'REPRODUCIBILITY SETUP' in source:
                print(f"✓ Reproducibility cell already exists in {notebook_path.name}")
                return True
    
    # Insert reproducibility cell after setup cell
    cells.insert(setup_idx + 1, REPRODUCIBILITY_CELL.copy())
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Added reproducibility cell to {notebook_path.name}")
    return True

if __name__ == '__main__':
    notebooks_dir = Path('notebooks')
    notebook_files = [
        '01_data_split.ipynb',
        '02_feature_extraction_separate.ipynb',
        '03_train_evaluate.ipynb',
        '03_5_ablation_study.ipynb',
        '04_model_specific_top15_fusion.ipynb',
        '05_final_evaluation.ipynb',
        'appendix_early_fusion.ipynb',
    ]
    
    print("Adding reproducibility setup cells to all notebooks...\n")
    
    for nb_name in notebook_files:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            add_reproducibility_cell(nb_path)
        else:
            print(f"⚠ Warning: {nb_name} not found")
    
    print("\n✓ Done! All notebooks now have reproducibility setup cells.")
    print("\nIMPORTANT: Make sure to run the reproducibility cell FIRST in each notebook!")

