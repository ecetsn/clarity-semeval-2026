"""
Storage manager - handles saving/loading features, models, results
GitHub for metadata, Drive for large data files
"""
import json
import numpy as np
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
            'features/raw', 'features/fused', 'features/probabilities',
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
        if not splits_path.exists():
            raise FileNotFoundError(
                f"Splits file not found: {splits_path}\n"
                f"Make sure you've run 01_data_split.ipynb for task '{task}' first."
            )
        
        # Load saved data
        with open(splits_path, 'rb') as f:
            splits_data = pickle.load(f)
        
        # NEW: If dataset data is saved in pickle (as lists of dicts), convert to SimpleDataset
        # This ensures no HuggingFace dependency is needed
        if split_name == 'train' and 'train_ds' in splits_data:
            data = splits_data['train_ds']
            # Check if it's already a list of dicts (new format)
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
        
        # BACKWARD COMPATIBILITY: If datasets not in pickle, this is an error
        # Old pickle files that only contain indices are no longer supported
        # because they require HuggingFace access which can timeout
        raise RuntimeError(
            f"Dataset split '{split_name}' not found in pickle file: {splits_path}\n"
            f"The pickle file appears to be in an old format that only contains indices.\n"
            f"Please re-run 01_data_split.ipynb to save dataset objects in the new format.\n"
            f"The new format saves full dataset data (as lists of dicts) and does not require HuggingFace."
        )
        
        # Get indices for requested split
        if split_name == 'train':
            indices = splits_data['train_indices']
            # Train and dev come from HuggingFace train split
            original_split = dataset['train']
        elif split_name == 'dev':
            indices = splits_data['dev_indices']
            # Train and dev come from HuggingFace train split
            original_split = dataset['train']
        elif split_name == 'test':
            indices = splits_data['test_indices']
            # Test comes from HuggingFace test split
            original_split = dataset['test']
        else:
            raise ValueError(f"Unknown split name: {split_name}. Must be 'train', 'dev', or 'test'")
        
        # Select indices from original split
        split_ds = original_split.select(indices.tolist())
        
        # For evasion: apply majority voting if not already done
        if task == 'evasion' and not is_filtered:
            # This shouldn't happen if splits were saved correctly, but handle it anyway
            split_ds = build_evasion_majority_dataset(split_ds, verbose=True)
        
        return split_ds
    
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
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
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
        """Load fused features"""
        model_str = '_'.join(sorted(models))
        npy_path = self.data_path / f'features/fused/X_{split}_fused_{model_str}_{task}.npy'
        if not npy_path.exists():
            raise FileNotFoundError(f"Fused features not found: {npy_path}")
        return np.load(npy_path)
    
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
            # Eğer use_paper_style=True ise, yeniden style_table_paper çağır (mapping ile)
            # Çünkü notebook'ta display için mapping=False ile oluşturulmuş olabilir
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
            from ..evaluation.tables import export_table_latex
            tex_path = save_dir / f'{table_name}.tex'
            export_table_latex(df_to_save, tex_path, caption=f"Results: {table_name.replace('_', ' ').title()}")
            saved_paths['tex'] = tex_path
        
        # Save Markdown
        if 'md' in formats or 'markdown' in formats:
            from ..evaluation.tables import export_table_markdown
            md_path = save_dir / f'{table_name}.md'
            export_table_markdown(df_to_save, md_path)
            saved_paths['md'] = md_path
        
        # Save PNG (requires matplotlib)
        if 'png' in formats:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                
                fig, ax = plt.subplots(figsize=(12, max(8, len(df_to_save) * 0.3)))
                ax.axis('tight')
                ax.axis('off')
                
                # Convert DataFrame to table
                table = ax.table(
                    cellText=df_to_save.values,
                    colLabels=df_to_save.columns,
                    rowLabels=df_to_save.index,
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

