"""
Data splitting utilities - splits HuggingFace train split into Train/Dev
"""
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple, Any


def split_train_into_train_dev(
    train_dataset,
    dev_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[Any, Any]:
    """
    Split HuggingFace train split into Train and Dev (80-20 split)
    
    Args:
        train_dataset: HuggingFace train split (from dataset['train'])
        dev_ratio: Ratio for dev set (0.20 = 20% of train data becomes dev)
        seed: Random seed for reproducibility
    
    Returns:
        train_ds, dev_ds
    """
    # Convert to indices for splitting
    if hasattr(train_dataset, 'select'):
        indices = np.arange(len(train_dataset))
    else:
        indices = np.arange(len(train_dataset))
    
    # Split train into train and dev (80-20)
    train_indices, dev_indices = train_test_split(
        indices,
        test_size=dev_ratio,  # 20% becomes dev
        random_state=seed,
        shuffle=True
    )
    
    # Create splits
    if hasattr(train_dataset, 'select'):
        train_ds = train_dataset.select(train_indices.tolist())
        dev_ds = train_dataset.select(dev_indices.tolist())
    else:
        train_ds = [train_dataset[i] for i in train_indices]
        dev_ds = [train_dataset[i] for i in dev_indices]
    
    print(f"Dataset split:")
    print(f"   Train: {len(train_ds)} samples ({len(train_indices)/len(indices)*100:.1f}%)")
    print(f"   Dev: {len(dev_ds)} samples ({len(dev_indices)/len(indices)*100:.1f}%)")
    
    return train_ds, dev_ds

