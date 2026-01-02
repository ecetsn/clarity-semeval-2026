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
        """Load features"""
        npy_path = self.data_path / f'features/raw/X_{split}_{model_name}_{task}.npy'
        if not npy_path.exists():
            raise FileNotFoundError(f"Features not found: {npy_path}")
        return np.load(npy_path)
    
    def save_predictions(self, predictions: np.ndarray, model_name: str, classifier: str, 
                        task: str, split: str) -> Path:
        """
        Save predictions (hard labels)
        - Large file → Drive
        - Metadata → GitHub
        """
        # Save predictions to Drive
        npy_path = self.data_path / f'predictions/pred_{split}_{model_name}_{classifier}_{task}.npy'
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
        
        meta_path = self.github_path / f'metadata/pred_{split}_{model_name}_{classifier}_{task}.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved predictions: {npy_path}")
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
        
        print(f"Saved probabilities: {npy_path}")
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
    
    def save_splits(self, train_ds, dev_ds, test_ds):
        """Save dataset splits"""
        import pickle
        
        # Save splits to Drive
        splits_path = self.data_path / 'splits/dataset_splits.pkl'
        with open(splits_path, 'wb') as f:
            pickle.dump({
                'train': train_ds,
                'dev': dev_ds,
                'test': test_ds
            }, f)
        
        # Save metadata to GitHub
        metadata = {
            "train_size": len(train_ds),
            "dev_size": len(dev_ds),
            "test_size": len(test_ds),
            "data_path": str(splits_path),
            "timestamp": datetime.now().isoformat()
        }
        
        meta_path = self.github_path / 'metadata/splits.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved splits: {splits_path}")
    
    def load_split(self, split_name: str):
        """Load dataset split"""
        import pickle
        splits_path = self.data_path / 'splits/dataset_splits.pkl'
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
        return splits[split_name]
    
    def save_results(self, results_dict: Dict, experiment_id: str) -> Path:
        """Save experiment results to GitHub"""
        results_path = self.github_path / f'results/{experiment_id}.json'
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Saved results: {results_path}")
        return results_path
    
    def load_metadata(self, model_name: str, task: str, split: str) -> Dict:
        """Load feature metadata"""
        meta_path = self.github_path / f'metadata/features_{split}_{model_name}_{task}.json'
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
        save_dir: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Save DataFrame table in multiple formats (CSV, HTML, PNG)
        
        Args:
            df: DataFrame or Styled DataFrame to save
            table_name: Name for the table file (without extension)
            formats: List of formats to save ('csv', 'html', 'png')
            save_dir: Directory to save (default: results/tables in data_path)
        
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
        else:
            df_to_save = df
        
        # Save CSV
        if 'csv' in formats:
            csv_path = save_dir / f'{table_name}.csv'
            df_to_save.to_csv(csv_path, index=True)
            saved_paths['csv'] = csv_path
            print(f"Saved table (CSV): {csv_path}")
        
        # Save HTML
        if 'html' in formats:
            html_path = save_dir / f'{table_name}.html'
            if hasattr(df, 'render'):
                # Styled DataFrame
                html_content = df.render()
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            else:
                # Regular DataFrame
                df_to_save.to_html(html_path, index=True, escape=False)
            saved_paths['html'] = html_path
            print(f"Saved table (HTML): {html_path}")
        
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
        filename: str = 'all_results.pkl'
    ) -> Path:
        """
        Save all_results dictionary to persistent storage
        
        Args:
            all_results: Complete results dictionary
            filename: Filename for saved dictionary
        
        Returns:
            Path to saved file
        """
        # Save to Drive (large file)
        pkl_path = self.data_path / f'results/{filename}'
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
            print(f"Saved all_results (JSON): {json_path}")
        except Exception as e:
            print(f"WARNING: Could not save JSON version: {e}")
        
        print(f"Saved all_results (Pickle): {pkl_path}")
        return pkl_path

