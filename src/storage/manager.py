"""
Storage manager - handles saving/loading features, models, results
GitHub for metadata, Drive for large data files
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pickle


class SimpleDataset:
    """
    Simple wrapper for list of dicts to mimic HuggingFace Dataset interface.
    This allows saved dataset splits to be used without HuggingFace dependency.
    """
    def __init__(self, data_list):
        """
        Args:
            data_list: List of dictionaries, where each dict represents a sample
        """
        self._data = data_list
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx_or_key):
        """
        Support both integer indexing (dataset[0]) and key-based column access (dataset['key']).
        This mimics HuggingFace Dataset behavior.
        """
        if isinstance(idx_or_key, (int, slice)):
            # Integer indexing: return single sample or slice
            return self._data[idx_or_key]
        elif isinstance(idx_or_key, str):
            # Key-based access: return column (list of values for that key)
            return [item.get(idx_or_key) for item in self._data]
        else:
            raise TypeError(f"Index must be int, slice, or str, got {type(idx_or_key)}")
    
    def __iter__(self):
        return iter(self._data)
    
    @property
    def column_names(self):
        """
        Return list of column names (keys from the first sample).
        Mimics HuggingFace Dataset.column_names property.
        """
        if len(self._data) > 0:
            return list(self._data[0].keys())
        return []
    
    def select(self, indices):
        """
        Select samples by indices (mimics HuggingFace Dataset.select).
        
        Args:
            indices: List of integer indices to select
        
        Returns:
            New SimpleDataset with selected samples
        """
        selected_data = [self._data[i] for i in indices]
        return SimpleDataset(selected_data)
    
    def remove_columns(self, column_names):
        """
        Remove columns from dataset (mimics HuggingFace Dataset.remove_columns).
        
        Args:
            column_names: List of column names to remove
        
        Returns:
            New SimpleDataset with columns removed
        """
        new_data = []
        for item in self._data:
            new_item = {k: v for k, v in item.items() if k not in column_names}
            new_data.append(new_item)
        return SimpleDataset(new_data)
    
    def add_column(self, name, column):
        """
        Add a new column to dataset (mimics HuggingFace Dataset.add_column).
        
        Args:
            name: Column name
            column: List of values (must match dataset length)
        
        Returns:
            New SimpleDataset with column added
        """
        if len(column) != len(self._data):
            raise ValueError(f"Column length {len(column)} doesn't match dataset length {len(self._data)}")
        
        new_data = []
        for i, item in enumerate(self._data):
            new_item = item.copy()
            new_item[name] = column[i]
            new_data.append(new_item)
        return SimpleDataset(new_data)


class StorageManager:
    """Manages storage: GitHub for code/metadata, Drive for large data"""
    
    def __init__(self, base_path: str, data_path: Optional[str] = None, github_path: Optional[str] = None):
        """
        Initialize storage manager
        
        Args:
            base_path: Base path (usually GitHub repo)
            data_path: Path for large data files (Drive or local)
            github_path: Path for metadata/results (GitHub repo)
        """
        self.base_path = Path(base_path)
        self.data_path = Path(data_path) if data_path else self.base_path
        self.github_path = Path(github_path) if github_path else self.base_path
        
        # Create directories for data (Drive)
        for dir_name in [
            'features/raw', 'features/fused', 'features/probabilities', 'features/model_independent',
            'models/classifiers', 'models/fusion',
            'predictions', 'results', 'checkpoints', 'splits'
        ]:
            (self.data_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create directories for metadata (GitHub)
        for dir_name in ['metadata', 'results']:
            (self.github_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    def save_features(self, X: np.ndarray, model_name: str, task: str, split: str, feature_names: List[str]) -> Path:
        """
        Save features with metadata
        
        Args:
            X: Feature matrix (N, F)
            model_name: Model name (e.g., 'bert', 'roberta')
            task: Task name ('clarity' or 'evasion')
            split: Split name ('train', 'dev', 'test')
            feature_names: List of feature names
        
        Returns:
            Path to saved numpy file
        """
        # Save numpy array to Drive (large file)
        npy_path = self.data_path / f'features/raw/X_{split}_{model_name}_{task}.npy'
        np.save(npy_path, X)
        
        # Save metadata to GitHub (small JSON)
        metadata = {
            "model": model_name,
            "task": task,
            "split": split,
            "type": "features",
            "shape": list(X.shape),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "data_path": str(npy_path),
            "timestamp": datetime.now().isoformat()
        }
        
        meta_path = self.github_path / f'metadata/features_{split}_{model_name}_{task}.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved features: {npy_path}")
        print(f"Saved metadata: {meta_path}")
        return npy_path
    
    def load_features(self, model_name: str, task: str, split: str) -> np.ndarray:
        """
        Load features from persistent storage
        
        Args:
            model_name: Model name (e.g., 'bert', 'roberta')
            task: Task name ('clarity' or 'evasion')
            split: Split name ('train', 'dev', 'test')
        
        Returns:
            Feature matrix (N, F) loaded from Google Drive
        
        Raises:
            FileNotFoundError: If feature file does not exist
        
        Note:
            - Features are stored in Google Drive (persistent across Colab runtimes)
            - File format: X_{split}_{model_name}_{task}.npy
            - This is NOT dependent on HuggingFace cache (unlike dataset splits)
        """
        npy_path = self.data_path / f'features/raw/X_{split}_{model_name}_{task}.npy'
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Features not found: {npy_path}\n"
                f"  Model: {model_name}, Task: {task}, Split: {split}\n"
                f"  Expected path: {npy_path}\n"
                f"  Make sure you have run feature extraction for this combination."
            )
        return np.load(npy_path)
    
    def list_saved_features(self) -> dict:
        """
        List all saved feature files (for verification)
        
        Returns:
            Dictionary mapping (model, task, split) -> file_path
        """
        features_dir = self.data_path / 'features/raw'
        if not features_dir.exists():
            return {}
        
        saved_features = {}
        for npy_file in features_dir.glob('X_*.npy'):
            # Parse filename: X_{split}_{model}_{task}.npy
            parts = npy_file.stem.split('_')
            if len(parts) < 4:
                continue
            split = parts[1]
            task = parts[-1]
            model = '_'.join(parts[2:-1])
            saved_features[(model, task, split)] = str(npy_file)
        
        return saved_features
    
    def save_model_independent_features(
        self, 
        X: np.ndarray, 
        split: str, 
        feature_names: List[str],
        task: str = 'clarity',  # Task-specific: clarity and evasion have different splits
        question_key: str = "question"
    ) -> Path:
        """
        Save model-independent features (1 kez çıkar, tüm modeller için kullan)
        Task-specific because clarity and evasion have different splits
        
        Args:
            X: Feature matrix (N, 18) - model-independent features
            split: Split name ('train', 'dev', 'test')
            feature_names: List of 18 feature names
            task: Task name ('clarity' or 'evasion') - determines which split file to save
            question_key: Question key used (for metadata)
        
        Returns:
            Path to saved numpy file
        """
        # Save numpy array to Drive (task-specific)
        npy_path = self.data_path / f'features/model_independent/X_{split}_{task}_independent.npy'
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, X)
        
        # Save metadata to GitHub
        metadata = {
            "type": "model_independent_features",
            "split": split,
            "task": task,
            "question_key": question_key,
            "shape": list(X.shape),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "data_path": str(npy_path),
            "timestamp": datetime.now().isoformat()
        }
        
        meta_path = self.github_path / f'metadata/features_independent_{split}_{task}.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved model-independent features: {npy_path}")
        print(f"Saved metadata: {meta_path}")
        return npy_path
    
    def load_model_independent_features(self, split: str, task: str = 'clarity') -> np.ndarray:
        """
        Load model-independent features (1 kez çıkar, tüm modeller için kullan)
        Task-specific because clarity and evasion have different splits
        
        Args:
            split: Split name ('train', 'dev', 'test')
            task: Task name ('clarity' or 'evasion') - determines which split file to load
        
        Returns:
            Feature matrix (N, 18) loaded from Google Drive
        
        Raises:
            FileNotFoundError: If feature file does not exist
        """
        npy_path = self.data_path / f'features/model_independent/X_{split}_{task}_independent.npy'
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Model-independent features not found: {npy_path}\n"
                f"  Split: {split}, Task: {task}\n"
                f"  Expected path: {npy_path}\n"
                f"  Make sure you have run model-independent feature extraction for this split and task."
            )
        return np.load(npy_path)
    
    def save_predictions(self, predictions: np.ndarray, model_name: str, classifier: str, 
                        task: str, split: str, save_dir: Optional[str] = None,
                        metadata_dir: Optional[str] = None) -> Path:
        """
        Save predictions (hard labels)
        - Large file → Drive
        - Metadata → GitHub
        
        Args:
            predictions: Predictions array
            model_name: Model name
            classifier: Classifier name
            task: Task name
            split: Split name ('train', 'dev', 'test')
            save_dir: Custom directory for predictions (default: predictions/)
            metadata_dir: Custom directory for metadata (default: metadata/)
        """
        # Save predictions to Drive
        if save_dir is None:
            npy_path = self.data_path / f'predictions/pred_{split}_{model_name}_{classifier}_{task}.npy'
        else:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            npy_path = save_dir_path / f'pred_{split}_{model_name}_{classifier}_{task}.npy'
        
        np.save(npy_path, predictions)
        
        # Save metadata to GitHub
        metadata = {
            "model": model_name,
            "classifier": classifier,
            "task": task,
            "split": split,
            "type": "hard_labels",
            "shape": list(predictions.shape),
            "n_classes": len(np.unique(predictions)),
            "data_path": str(npy_path),
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata_dir is None:
            meta_path = self.github_path / f'metadata/pred_{split}_{model_name}_{classifier}_{task}.json'
        else:
            metadata_dir_path = self.github_path / metadata_dir
            metadata_dir_path.mkdir(parents=True, exist_ok=True)
            meta_path = metadata_dir_path / f'pred_{split}_{model_name}_{classifier}_{task}.json'
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return npy_path
    
    def save_probabilities(self, probabilities: np.ndarray, model_name: str, classifier: str,
                          task: str, split: str) -> Path:
        """
        Save probabilities (for late fusion)
        - Large file → Drive
        - Metadata → GitHub
        """
        # Save probabilities to Drive
        npy_path = self.data_path / f'features/probabilities/probs_{split}_{model_name}_{classifier}_{task}.npy'
        np.save(npy_path, probabilities)
        
        # Save metadata to GitHub
        metadata = {
            "model": model_name,
            "classifier": classifier,
            "task": task,
            "split": split,
            "type": "probabilities",
            "shape": list(probabilities.shape),
            "n_classes": probabilities.shape[1],
            "data_path": str(npy_path),
            "timestamp": datetime.now().isoformat()
        }
        
        meta_path = self.github_path / f'metadata/probs_{split}_{model_name}_{classifier}_{task}.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return npy_path
    
    def save_fused_features(self, X: np.ndarray, models: List[str], task: str, split: str,
                           feature_names: List[str], fusion_method: str = 'concat') -> Path:
        """Save fused features"""
        model_str = '_'.join(models)
        npy_path = self.data_path / f'features/fused/X_{split}_fused_{model_str}_{task}.npy'
        np.save(npy_path, X)
        
        metadata = {
            "models": models,
            "task": task,
            "split": split,
            "fusion_method": fusion_method,
            "type": "fused_features",
            "shape": list(X.shape),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "data_path": str(npy_path),
            "timestamp": datetime.now().isoformat()
        }
        
        meta_path = self.github_path / f'metadata/fused_{split}_{model_str}_{task}.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved fused features: {npy_path}")
        return npy_path
    
    def save_splits(self, train_ds, dev_ds, test_ds, train_raw=None, dev_ratio: float = 0.20, seed: int = 42, dataset_name: str = "ailsntua/QEvasion", task: str = "clarity"):
        """
        Save dataset splits by storing indices instead of dataset objects
        This makes splits runtime-independent (no HuggingFace cache dependencies)
        
        IMPORTANT: Clarity and Evasion have different splits because Evasion uses majority voting
        which drops samples without strict majority. Therefore, splits are saved per task.
        
        Args:
            train_ds: Train split dataset (from train_raw, possibly filtered for evasion)
            dev_ds: Dev split dataset (from train_raw, possibly filtered for evasion)
            test_ds: Test split dataset (from HuggingFace test split, possibly filtered for evasion)
            train_raw: Original HuggingFace train split (required to find indices)
            dev_ratio: Dev ratio used for splitting (default: 0.20)
            seed: Random seed used for splitting (default: 42)
            dataset_name: HuggingFace dataset name (default: "ailsntua/QEvasion")
            task: Task name ('clarity' or 'evasion') - determines which split file to save
        """
        import pickle
        from sklearn.model_selection import train_test_split
        
        if task not in ['clarity', 'evasion']:
            raise ValueError(f"Task must be 'clarity' or 'evasion', got '{task}'")
        
        # For test_ds: get actual indices from original test split
        # We need to map filtered test_ds back to original test indices
        # For now, we'll save the filtered dataset's relative indices
        # This requires knowing which samples were kept after filtering
        
        # For train_ds and dev_ds: if they're filtered (evasion), we need to track
        # which original indices they correspond to
        
        # Save indices and metadata to Drive (task-specific)
        splits_path = self.data_path / f'splits/dataset_splits_{task}.pkl'
        
        # For clarity: normal split indices
        # For evasion: we need to track which samples were kept after majority voting
        if task == 'clarity':
            # Normal split - recreate indices
            if train_raw is None:
                raise ValueError("train_raw is required to save splits. Please provide the original HuggingFace train split.")
            
            indices = np.arange(len(train_raw))
            train_indices, dev_indices = train_test_split(
                indices,
                test_size=dev_ratio,
                random_state=seed,
                shuffle=True
            )
            
            # Test indices (full test split)
            test_indices = np.arange(len(test_ds))
            
            # Verify sizes match
            if len(train_indices) != len(train_ds) or len(dev_indices) != len(dev_ds):
                raise ValueError(
                    f"Split sizes don't match! "
                    f"Expected train: {len(train_ds)}, got {len(train_indices)}; "
                    f"Expected dev: {len(dev_ds)}, got {len(dev_indices)}"
                )
        else:  # task == 'evasion'
            # Evasion: filtered dataset - we need to save the actual sample indices
            # that were kept after majority voting
            # For now, save as sequential indices (0, 1, 2, ...) and mark as filtered
            # The actual mapping will be handled by the dataset itself
            
            # We'll save the dataset's internal indices
            # This is a simplified approach - in practice, we'd need to track
            # which original indices correspond to filtered samples
            train_indices = np.arange(len(train_ds))
            dev_indices = np.arange(len(dev_ds))
            test_indices = np.arange(len(test_ds))
        
        # Convert HuggingFace Dataset objects to lists of dicts for proper serialization
        # This ensures all data is saved and no HuggingFace dependency is needed
        print(f"  Converting datasets to dict format for serialization...")
        train_ds_dict = [train_ds[i] for i in range(len(train_ds))]
        dev_ds_dict = [dev_ds[i] for i in range(len(dev_ds))]
        test_ds_dict = [test_ds[i] for i in range(len(test_ds))]
        
        with open(splits_path, 'wb') as f:
            pickle.dump({
                'dataset_name': dataset_name,
                'task': task,
                'train_indices': train_indices,
                'dev_indices': dev_indices,
                'test_indices': test_indices,
                'dev_ratio': dev_ratio,
                'seed': seed,
                'train_size': len(train_ds),
                'dev_size': len(dev_ds),
                'test_size': len(test_ds),
                'is_filtered': (task == 'evasion'),  # Mark if this is a filtered dataset
                # Save actual dataset data as lists of dicts to avoid HuggingFace dependency
                # This ensures all data is fully serialized and can be loaded without HuggingFace
                'train_ds': train_ds_dict,
                'dev_ds': dev_ds_dict,
                'test_ds': test_ds_dict
            }, f)
        
        # Save metadata to GitHub
        metadata = {
            "dataset_name": dataset_name,
            "task": task,
            "train_size": len(train_ds),
            "dev_size": len(dev_ds),
            "test_size": len(test_ds),
            "is_filtered": (task == 'evasion'),
            "data_path": str(splits_path),
            "timestamp": datetime.now().isoformat()
        }
        
        meta_path = self.github_path / f'metadata/splits_{task}.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved splits (indices) for task '{task}': {splits_path}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Train: {len(train_ds)} samples")
        print(f"  Dev: {len(dev_ds)} samples")
        print(f"  Test: {len(test_ds)} samples")
        if task == 'evasion':
            print(f"  NOTE: Evasion splits are filtered (majority voting applied)")
    
    def load_split(self, split_name: str, task: str = 'clarity'):
        """
        Load dataset split from pickle file (no HuggingFace dependency)
        This makes splits runtime-independent (no HuggingFace cache dependencies)
        
        IMPORTANT: Clarity and Evasion have different splits because Evasion uses majority voting.
        Always specify the task when loading splits.
        
        Args:
            split_name: Split name ('train', 'dev', 'test')
            task: Task name ('clarity' or 'evasion') - determines which split file to load
        
        Returns:
            Dataset split loaded from pickle file (or from HuggingFace if pickle doesn't contain datasets - backward compatibility)
        """
        import pickle
        from datasets import load_dataset
        from src.data.splitter import build_evasion_majority_dataset
        
        if task not in ['clarity', 'evasion']:
            raise ValueError(f"Task must be 'clarity' or 'evasion', got '{task}'")
        
        splits_path = self.data_path / f'splits/dataset_splits_{task}.pkl'
        
        # Debug: Check if data_path exists and is accessible
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data path does not exist: {self.data_path}\n"
                f"  This usually means Google Drive is not mounted correctly.\n"
                f"  Solution: Make sure you've run 'drive.mount('/content/drive')' in your notebook."
            )
        
        # Check if splits directory exists
        splits_dir = self.data_path / 'splits'
        if not splits_dir.exists():
            raise FileNotFoundError(
                f"Splits directory does not exist: {splits_dir}\n"
                f"  Data path: {self.data_path}\n"
                f"  Solution: Run 01_data_split.ipynb first to create split files."
            )
        
        if not splits_path.exists():
            # Check for alternative file names (backward compatibility)
            alt_paths = [
                self.data_path / f'splits/dataset_splits.pkl',  # Old format (no task suffix)
                self.data_path / 'splits' / f'dataset_splits_{task}.pkl',  # Explicit path
            ]
            
            found_alt = None
            for alt_path in alt_paths:
                if alt_path.exists():
                    found_alt = alt_path
                    break
            
            if found_alt:
                # Use alternative path but warn user
                print(f"  Warning: Using alternative split file: {found_alt}")
                print(f"  Expected: {splits_path}")
                splits_path = found_alt
            else:
                # Provide detailed error message with debugging info
                available_files = []
                if splits_dir.exists():
                    available_files = [f.name for f in splits_dir.glob('*.pkl')]
                    # Also check absolute path
                    try:
                        abs_path = splits_path.resolve()
                        abs_exists = abs_path.exists()
                    except Exception:
                        abs_exists = False
                        abs_path = None
                
                error_msg = (
                f"Splits file not found: {splits_path}\n"
                    f"  Task: {task}\n"
                    f"  Data path: {self.data_path}\n"
                    f"  Data path exists: {self.data_path.exists()}\n"
                    f"  Splits directory: {splits_dir}\n"
                    f"  Splits directory exists: {splits_dir.exists()}\n"
                    f"  Expected file: splits/dataset_splits_{task}.pkl\n"
                )
                if abs_path:
                    error_msg += (
                        f"  Absolute path: {abs_path}\n"
                        f"  Absolute path exists: {abs_exists}\n"
                    )
                if available_files:
                    error_msg += (
                        f"  Available split files in splits/ directory:\n"
                    )
                    for f in available_files:
                        error_msg += f"    - {f}\n"
                else:
                    error_msg += (
                        f"  No .pkl files found in splits/ directory.\n"
                        f"  This might indicate:\n"
                        f"    1. Drive is not mounted correctly\n"
                        f"    2. Files are in a different location\n"
                        f"    3. Split files were not created yet\n"
                    )
                error_msg += (
                    f"  Solution: Run 01_data_split.ipynb for task '{task}' first.\n"
                    f"  The split file should be saved as: splits/dataset_splits_{task}.pkl"
            )
                raise FileNotFoundError(error_msg)
        
        # Load saved data
        with open(splits_path, 'rb') as f:
            splits_data = pickle.load(f)
        
        # If dataset data is saved in pickle (as lists of dicts), convert to SimpleDataset
        # This ensures no HuggingFace dependency is needed
        if split_name == 'train' and 'train_ds' in splits_data:
            data = splits_data['train_ds']
            # Check if it's already a list of dicts
            if isinstance(data, list):
                return SimpleDataset(data)
            else:
                # Old format: Dataset object (may not be fully serialized)
                # Convert to list of dicts to avoid HuggingFace dependency
                print(f"  Warning: Found Dataset object in pickle (old format). Converting to dict list...")
                data_list = [data[i] for i in range(len(data))]
                return SimpleDataset(data_list)
        elif split_name == 'dev' and 'dev_ds' in splits_data:
            data = splits_data['dev_ds']
            if isinstance(data, list):
                return SimpleDataset(data)
            else:
                # Old format: Convert to list of dicts
                print(f"  Warning: Found Dataset object in pickle (old format). Converting to dict list...")
                data_list = [data[i] for i in range(len(data))]
                return SimpleDataset(data_list)
        elif split_name == 'test' and 'test_ds' in splits_data:
            data = splits_data['test_ds']
            if isinstance(data, list):
                return SimpleDataset(data)
            else:
                # Old format: Convert to list of dicts
                print(f"  Warning: Found Dataset object in pickle (old format). Converting to dict list...")
                data_list = [data[i] for i in range(len(data))]
                return SimpleDataset(data_list)
        
        # If we reach here, the split was not found in pickle file
        # This means the pickle file is in an old format that only contains indices
        # Old format is no longer supported because it requires HuggingFace access which can timeout
        raise RuntimeError(
            f"Dataset split '{split_name}' not found in pickle file: {splits_path}\n"
            f"The pickle file appears to be in an old format that only contains indices.\n"
            f"Please re-run 01_data_split.ipynb to save dataset objects in the new format.\n"
            f"The new format saves full dataset data (as lists of dicts) and does not require HuggingFace."
        )
    
    def save_results(self, results_dict: Dict, experiment_id: str, save_dir: Optional[str] = None) -> Path:
        """
        Save experiment results to GitHub
        
        Args:
            results_dict: Results dictionary
            experiment_id: Experiment ID (filename without extension)
            save_dir: Custom directory for results (default: results/)
        """
        if save_dir is None:
            results_path = self.github_path / f'results/{experiment_id}.json'
        else:
            save_dir_path = self.github_path / save_dir
            save_dir_path.mkdir(parents=True, exist_ok=True)
            results_path = save_dir_path / f'{experiment_id}.json'
        
        # FIX: Convert numpy arrays and types to JSON-serializable Python types
        # NumPy 2.0 compatible: use np.floating and np.integer instead of deprecated np.float_, np.int_
        def make_json_serializable(obj):
            """Recursively convert numpy arrays and types to JSON-serializable Python types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):  # Covers all integer types (int8, int16, int32, int64, etc.)
                return int(obj)
            elif isinstance(obj, np.floating):  # Covers all float types (float16, float32, float64, etc.)
                return float(obj)
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar (fallback for any numpy scalar type)
                return obj.item()
            else:
                return obj
        
        # Convert results_dict to JSON-serializable format
        results_serializable = make_json_serializable(results_dict)
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        return results_path
    
    def load_metadata(self, model_name: str, task: str, split: str) -> Dict:
        """Load feature metadata"""
        meta_path = self.github_path / f'metadata/features_{split}_{model_name}_{task}.json'
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {meta_path}\n"
                f"  Model: {model_name}, Task: {task}, Split: {split}\n"
                f"  Make sure you have run feature extraction (02_feature_extraction_separate.ipynb) "
                f"and metadata files are present in the repository."
            )
        with open(meta_path, 'r') as f:
            return json.load(f)
    
    def load_predictions(self, model_name: str, classifier: str, task: str, split: str) -> np.ndarray:
        """Load predictions (hard labels)"""
        npy_path = self.data_path / f'predictions/pred_{split}_{model_name}_{classifier}_{task}.npy'
        if not npy_path.exists():
            raise FileNotFoundError(f"Predictions not found: {npy_path}")
        return np.load(npy_path)
    
    def load_probabilities(self, model_name: str, classifier: str, task: str, split: str) -> np.ndarray:
        """Load probabilities"""
        npy_path = self.data_path / f'features/probabilities/probs_{split}_{model_name}_{classifier}_{task}.npy'
        if not npy_path.exists():
            raise FileNotFoundError(f"Probabilities not found: {npy_path}")
        return np.load(npy_path)
    
    def load_fused_features(self, models: List[str], task: str, split: str) -> np.ndarray:
        """Load fused features
        
        Tries both original order and sorted order for maximum compatibility.
        - First tries original order (matches save_fused_features which uses '_'.join(models))
        - If not found, tries sorted order (for backward compatibility with old files)
        
        This ensures compatibility with:
        - Notebook 4_5 (uses original order)
        - Any old files saved with sorted order
        - Future files saved with original order
        """
        # Try original order first (matches save_fused_features)
        model_str_original = '_'.join(models)
        npy_path_original = self.data_path / f'features/fused/X_{split}_fused_{model_str_original}_{task}.npy'
        
        if npy_path_original.exists():
            return np.load(npy_path_original)
        
        # Fallback: Try sorted order (for backward compatibility)
        model_str_sorted = '_'.join(sorted(models))
        npy_path_sorted = self.data_path / f'features/fused/X_{split}_fused_{model_str_sorted}_{task}.npy'
        
        if npy_path_sorted.exists():
            # Silent fallback - no warning needed, this is expected behavior
            return np.load(npy_path_sorted)
        
        # Neither found - raise error with both paths for debugging
        raise FileNotFoundError(
            f"Fused features not found for {task} (split: {split}).\n"
            f"  Tried original order: {npy_path_original}\n"
            f"  Tried sorted order: {npy_path_sorted}\n"
            f"  Models: {models}\n"
            f"  Please ensure fused features have been created (e.g., run STEP 4 in early fusion notebook)."
        )
    
    def load_results(self, experiment_id: str) -> Dict:
        """Load experiment results"""
        results_path = self.github_path / f'results/{experiment_id}.json'
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def save_table(
        self,
        df: Any,
        table_name: str,
        formats: List[str] = ['csv', 'html'],
        save_dir: Optional[str] = None,
        use_paper_style: bool = False
    ) -> Dict[str, Path]:
        """
        Save DataFrame table in multiple formats (CSV, HTML, PNG, LaTeX, Markdown)
        
        Args:
            df: DataFrame or Styled DataFrame to save
            table_name: Name for the table file (without extension)
            formats: List of formats to save ('csv', 'html', 'png', 'tex', 'md')
            save_dir: Directory to save (default: results/tables in data_path)
            use_paper_style: If True, use paper-ready styling (minimal, professional)
        
        Returns:
            Dict mapping format -> Path to saved file
        """
        if save_dir is None:
            save_dir = self.data_path / 'results/tables'
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Extract DataFrame if it's a Styler
        if hasattr(df, 'data'):
            df_to_save = df.data
            # If use_paper_style=True, reapply style_table_paper with mapping
            # Notebook display may have been created with mapping=False
            if use_paper_style:
                from ..evaluation.tables import style_table_paper
                # Pass table_name to style_table_paper for auto-detection of best_direction
                styled_df = style_table_paper(df_to_save, apply_column_mapping=True, table_name=table_name)
            else:
                styled_df = df  # Keep original styled version
        else:
            df_to_save = df
            styled_df = None
        
        # Save CSV
        if 'csv' in formats:
            csv_path = save_dir / f'{table_name}.csv'
            df_to_save.to_csv(csv_path, index=True)
            saved_paths['csv'] = csv_path
            print(f"Saved table (CSV): {csv_path}")
        
        # Save HTML
        if 'html' in formats:
            html_path = save_dir / f'{table_name}.html'
            if use_paper_style and styled_df is None:
                # Apply paper-ready styling WITH column mapping for saved files
                from ..evaluation.tables import style_table_paper
                # Pass table_name to style_table_paper for auto-detection of best_direction
                styled_df = style_table_paper(df_to_save, apply_column_mapping=True, table_name=table_name)
            
            if styled_df is not None and hasattr(styled_df, 'render'):
                # Paper-ready styled DataFrame
                html_content = styled_df.render()
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            elif hasattr(df, 'render'):
                # Original styled DataFrame
                html_content = df.render()
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            else:
                # Regular DataFrame
                df_to_save.to_html(html_path, index=True, escape=False)
            saved_paths['html'] = html_path
            print(f"Saved table (HTML): {html_path}")
        
        # Save LaTeX
        if 'tex' in formats or 'latex' in formats:
            try:
                from ..evaluation.tables import export_table_latex
                tex_path = save_dir / f'{table_name}.tex'
                export_table_latex(df_to_save, tex_path, caption=f"Results: {table_name.replace('_', ' ').title()}")
                saved_paths['tex'] = tex_path
            except Exception as e:
                print(f"WARNING: Could not save LaTeX table: {e}")
                import traceback
                traceback.print_exc()
        
        # Save Markdown
        if 'md' in formats or 'markdown' in formats:
            try:
                from ..evaluation.tables import export_table_markdown
                md_path = save_dir / f'{table_name}.md'
                export_table_markdown(df_to_save, md_path)
                saved_paths['md'] = md_path
            except Exception as e:
                print(f"WARNING: Could not save Markdown table: {e}")
                import traceback
                traceback.print_exc()
        
        # Save PNG (requires matplotlib)
        if 'png' in formats:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                
                # Replace NaN values with empty strings for display
                df_for_png = df_to_save.fillna("")
                
                fig, ax = plt.subplots(figsize=(12, max(8, len(df_for_png) * 0.3)))
                ax.axis('tight')
                ax.axis('off')
                
                # Convert DataFrame to table - handle NaN values
                cell_text = []
                for row in df_for_png.values:
                    cell_text.append([str(val) if pd.notna(val) and val != "" else "" for val in row])
                
                table = ax.table(
                    cellText=cell_text,
                    colLabels=[str(col) for col in df_for_png.columns],
                    rowLabels=[str(idx) for idx in df_for_png.index],
                    cellLoc='center',
                    loc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                
                png_path = save_dir / f'{table_name}.png'
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()
                saved_paths['png'] = png_path
                print(f"Saved table (PNG): {png_path}")
            except Exception as e:
                print(f"WARNING: Could not save PNG: {e}")
                import traceback
                traceback.print_exc()
        
        return saved_paths
    
    def save_all_results_dict(
        self,
        all_results: Dict,
        filename: str = 'all_results.pkl',
        save_dir: Optional[str] = None
    ) -> Path:
        """
        Save all_results dictionary to persistent storage
        
        Args:
            all_results: Complete results dictionary
            filename: Filename for saved dictionary
            save_dir: Custom directory for results dict (default: results/)
        
        Returns:
            Path to saved file
        """
        # Save to Drive (large file)
        if save_dir is None:
            pkl_path = self.data_path / f'results/{filename}'
        else:
            save_dir_path = self.data_path / save_dir
            save_dir_path.mkdir(parents=True, exist_ok=True)
            pkl_path = save_dir_path / filename
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        # Also save JSON version (for smaller size, but may lose some data)
        json_path = self.github_path / f'results/{filename.replace(".pkl", ".json")}'
        try:
            # Convert to JSON-serializable format
            json_results = json.loads(json.dumps(all_results, default=str))
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
        except Exception as e:
            pass  # Silently skip JSON save if it fails
        
        return pkl_path
    
    def load_all_results_dict(
        self,
        filename: str = 'all_results_dev.pkl'
    ) -> Dict:
        """
        Load all_results dictionary from persistent storage
        
        Args:
            filename: Filename of saved dictionary
            
        Returns:
            Complete results dictionary, or empty dict if not found
        """
        import pickle
        
        # Try to load from Drive (pkl file)
        pkl_path = self.data_path / f'results/{filename}'
        
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    all_results = pickle.load(f)
                return all_results
            except Exception as e:
                print(f"Warning: Could not load {pkl_path}: {e}")
                return {}
        else:
            return {}
    
    def save_tfidf_vectorizer(self, vectorizer: Any, task: str) -> Path:
        """
        Save TF-IDF vectorizer to cache (task-specific)
        
        Args:
            vectorizer: Fitted TF-IDF vectorizer
            task: Task name ('clarity' or 'evasion')
        
        Returns:
            Path to saved pickle file
        
        Raises:
            ValueError: If task is not 'clarity' or 'evasion'
            TypeError: If vectorizer is None
        """
        import pickle
        
        # Validate task
        if task not in ['clarity', 'evasion']:
            raise ValueError(f"Task must be 'clarity' or 'evasion', got '{task}'")
        
        # Validate vectorizer
        if vectorizer is None:
            raise TypeError("Cannot save None vectorizer to cache")
        
        cache_path = self.data_path / f'cache/tfidf_vectorizer_{task}.pkl'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"Saved TF-IDF vectorizer to cache: {cache_path}")
            return cache_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to save TF-IDF vectorizer to cache: {e}\n"
                f"  Cache path: {cache_path}\n"
                f"  Task: {task}"
            ) from e
    
    def load_tfidf_vectorizer(self, task: str) -> Optional[Any]:
        """
        Load TF-IDF vectorizer from cache (task-specific)
        
        Args:
            task: Task name ('clarity' or 'evasion')
        
        Returns:
            Fitted TF-IDF vectorizer, or None if not found or corrupted
        
        Note:
            Returns None if:
            - Cache file does not exist
            - Cache file is corrupted (pickle load fails)
            - Vectorizer is None or invalid
            
            This allows graceful fallback to fitting a new vectorizer.
        """
        import pickle
        
        # Validate task
        if task not in ['clarity', 'evasion']:
            print(f"Warning: Invalid task '{task}', returning None")
            return None
        
        cache_path = self.data_path / f'cache/tfidf_vectorizer_{task}.pkl'
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                vectorizer = pickle.load(f)
            
            # Validate loaded vectorizer
            if vectorizer is None:
                print(f"Warning: Loaded vectorizer is None from cache: {cache_path}")
                return None
            
            # Check if vectorizer has required methods (basic validation)
            if not hasattr(vectorizer, 'transform') or not hasattr(vectorizer, 'fit'):
                print(f"Warning: Loaded object is not a valid vectorizer: {cache_path}")
                return None
            
            print(f"Loaded TF-IDF vectorizer from cache: {cache_path}")
            return vectorizer
            
        except (pickle.PickleError, EOFError, ImportError) as e:
            # Corrupted pickle file or import error
            print(f"Warning: Could not load TF-IDF vectorizer from cache (corrupted?): {e}")
            print(f"  Cache path: {cache_path}")
            print(f"  Will fit new vectorizer instead")
            return None
        except Exception as e:
            # Other unexpected errors
            print(f"Warning: Unexpected error loading TF-IDF vectorizer from cache: {e}")
            print(f"  Cache path: {cache_path}")
            print(f"  Will fit new vectorizer instead")
            return None

