"""
Final Evaluation Type 2: Weighted Average Ensemble from Method 2 Models

This module implements weighted average ensemble using trained models from
04_model_specific_top15_fusion.ipynb (Method 2: Greedy-based top-15 features).
"""
import numpy as np
import torch
import pandas as pd
import json
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from .hierarchical import evaluate_hierarchical_approach
from ..storage.manager import StorageManager
from ..features.extraction import featurize_hf_dataset_in_batches_v2
from ..evaluation.metrics import compute_all_metrics, print_classification_report
from ..evaluation.tables import print_results_table, style_table, style_table_paper
from ..evaluation.visualizer import visualize_all_evaluation


def run_final_evaluation_type2(
    storage: StorageManager,
    models: List[str],
    tasks: List[str],
    model_configs: Dict[str, str],
    model_max_lengths: Dict[str, int],
    label_lists: Dict[str, List[str]],
    device: Optional[torch.device] = None,
    random_state: int = 42,
    batch_size: int = 8,
    save_results: bool = True,
    create_plots: bool = True,
) -> Dict[str, Any]:
    """
    Final Evaluation Type 2: Weighted Average Ensemble
    
    This function:
    1. Loads trained models from Method 2 (04_model_specific_top15_fusion.ipynb)
    2. Extracts or loads test features
    3. Predicts probabilities for each model×task×classifier combination
    4. Performs weighted average ensemble (weighted by dev set macro F1)
    5. Evaluates on test set (clarity, evasion, hierarchical, annotator-specific)
    6. Saves results to FinalResultsType2
    
    Args:
        storage: StorageManager instance
        models: Model listesi (e.g., ['bert', 'roberta', 'deberta', 'xlnet'])
        tasks: Task listesi (e.g., ['clarity', 'evasion'])
        model_configs: Model isimleri mapping (e.g., {'bert': 'bert-base-uncased'})
        model_max_lengths: Max sequence length mapping (e.g., {'bert': 512})
        label_lists: Label listeleri (e.g., {'clarity': ['Ambivalent', ...], 'evasion': [...]})
        device: torch device (None ise otomatik seçilir)
        random_state: Random seed
        batch_size: Feature extraction batch size
        save_results: Sonuçları kaydet (True/False)
        create_plots: Plot'ları oluştur (True/False)
    
    Returns:
        Dictionary with final results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========================================================================
    # CRITICAL: Create ALL output directories upfront (CHECKPOINT)
    # ========================================================================
    plots_dir = storage.data_path / 'results/FinalResultsType2/plots'
    predictions_dir = storage.data_path / 'results/FinalResultsType2/predictions'
    tables_dir = storage.data_path / 'results/FinalResultsType2/tables'
    results_dir = storage.data_path / 'results/FinalResultsType2'
    
    # GitHub directories (for metadata: JSON results)
    metadata_results_dir = storage.github_path / 'results/FinalResultsType2Results'
    
    # Create all directories (always, to prevent any FileNotFoundError)
    plots_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_results_dir.mkdir(parents=True, exist_ok=True)
    
    print("Created all output directories:")
    print(f"  Drive: {plots_dir}")
    print(f"  Drive: {predictions_dir}")
    print(f"  Drive: {tables_dir}")
    print(f"  Drive: {results_dir}")
    print(f"  GitHub: {metadata_results_dir}")
    
    # ========================================================================
    # STEP 1: Load Method 2 metadata
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOAD METHOD 2 METADATA")
    print("="*80)
    
    method2_metadata_path = storage.data_path / 'results/method2_trained_models.json'
    if not method2_metadata_path.exists():
        raise FileNotFoundError(
            f"Method 2 metadata not found: {method2_metadata_path}\n"
            f"Make sure you have run 04_model_specific_top15_fusion.ipynb first."
        )
    
    with open(method2_metadata_path, 'r') as f:
        method2_metadata = json.load(f)
    
    print(f"✓ Loaded Method 2 metadata from {method2_metadata_path}")
    
    # ========================================================================
    # STEP 2: Extract or load test features
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: TEST FEATURE EXTRACTION")
    print("="*80)
    
    for model_key in models:
        print(f"\n{'-'*60}")
        print(f"Model: {model_key.upper()}")
        print(f"{'-'*60}")
        
        for task in tasks:
            print(f"  Task: {task}")
            
            # Try to load existing features
            try:
                X_test = storage.load_features(model_key, task, 'test')
                print(f"    ✓ Loaded from Drive: {X_test.shape}")
            except FileNotFoundError:
                # Extract test features
                print(f"    → Test features not found in Drive. Extracting and saving...")
                
                # Load splits (required for feature extraction)
                try:
                    test_ds = storage.load_split('test', task=task)
                    train_ds = storage.load_split('train', task=task)  # For TF-IDF fitting
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"Split files not found. {e}\n"
                        f"Make sure you've run 01_data_split.ipynb for task '{task}' first."
                    )
                
                # Get model config
                model_name = model_configs[model_key]
                max_length = model_max_lengths[model_key]
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to(device)
                
                print(f"    Max sequence length: {max_length}")
                
                # Extract features
                X_test, _, _ = featurize_hf_dataset_in_batches_v2(
                    test_ds,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_sequence_length=max_length,
                    batch_size=batch_size,
                    task=task
                )
                
                # Save features
                storage.save_features(X_test, model_key, task, 'test')
                print(f"    ✓ Extracted and saved: {X_test.shape}")
    
    # ========================================================================
    # STEP 3: Load trained models and predict probabilities
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: LOAD MODELS AND PREDICT PROBABILITIES")
    print("="*80)
    
    models_fusion_dir = storage.data_path / 'models/fusion'
    
    # Store all probabilities and weights for ensemble
    ensemble_data = {}  # {task: {model: {classifier: (proba, weight)}}}
    
    for model_key in models:
        if model_key not in method2_metadata:
            print(f"⚠ Warning: Model '{model_key}' not found in Method 2 metadata")
            continue
        
        ensemble_data[model_key] = {}
        
        for task in tasks:
            if task not in method2_metadata[model_key]:
                print(f"⚠ Warning: Task '{task}' not found for model '{model_key}'")
                continue
            
            # Load test features
            X_test_full = storage.load_features(model_key, task, 'test')
            
            # Load test split for labels
            test_ds = storage.load_split('test', task=task)
            if task == 'clarity':
                label_key = 'clarity_label'
            else:  # evasion
                label_key = 'evasion_label'
            y_test = np.array([test_ds[i][label_key] for i in range(len(test_ds))])
            
            ensemble_data[model_key][task] = {}
            
            # Process each classifier for this model×task
            for clf_name, clf_metadata in method2_metadata[model_key][task].items():
                print(f"\n  {model_key} - {task} - {clf_name}:")
                
                # Load trained model
                model_filename = f"method2_{model_key}_{task}_{clf_name}.pkl"
                model_path = models_fusion_dir / model_filename
                
                if not model_path.exists():
                    print(f"    ⚠ Model not found: {model_path}")
                    continue
                
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                trained_model = model_data['model']
                selected_indices = model_data['selected_indices']
                label_list = model_data['label_list']
                label_encoder = model_data['label_encoder']
                
                # Extract selected features
                X_test = X_test_full[:, selected_indices]
                
                # Predict probabilities
                test_proba = trained_model.predict_proba(X_test)
                
                # Get weight (dev set macro F1)
                weight = clf_metadata.get('macro_f1_dev', 0.0)
                
                ensemble_data[model_key][task][clf_name] = {
                    'probabilities': test_proba,
                    'weight': weight,
                    'label_list': label_list,
                    'label_encoder': label_encoder
                }
                
                print(f"    ✓ Predicted probabilities: {test_proba.shape}")
                print(f"    ✓ Weight (dev F1): {weight:.4f}")
    
    # ========================================================================
    # STEP 4: Weighted Average Ensemble
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: WEIGHTED AVERAGE ENSEMBLE")
    print("="*80)
    
    final_results = {}
    
    for task in tasks:
        print(f"\n{'-'*60}")
        print(f"Task: {task.upper()}")
        print(f"{'-'*60}")
        
        label_list = label_lists[task]
        
        # Collect all probabilities and weights for this task
        probabilities_list = []
        weights_list = []
        
        for model_key in models:
            if model_key not in ensemble_data:
                continue
            if task not in ensemble_data[model_key]:
                continue
            
            for clf_name, clf_data in ensemble_data[model_key][task].items():
                proba = clf_data['probabilities']
                weight = clf_data['weight']
                
                # Verify label order matches
                if clf_data['label_list'] != label_list:
                    print(f"  ⚠ Warning: Label mismatch for {model_key}-{clf_name}")
                    continue
                
                probabilities_list.append(proba)
                weights_list.append(weight)
        
        if len(probabilities_list) == 0:
            print(f"  ⚠ No probabilities found for task '{task}'")
            continue
        
        # Normalize weights
        total_weight = sum(weights_list)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights_list]
        else:
            normalized_weights = [1.0 / len(weights_list)] * len(weights_list)
        
        # Weighted average ensemble
        ensemble_proba = np.zeros_like(probabilities_list[0])
        for proba, weight in zip(probabilities_list, normalized_weights):
            ensemble_proba += weight * proba
        
        # Get predictions
        ensemble_pred_indices = np.argmax(ensemble_proba, axis=1)
        ensemble_pred = np.array([label_list[i] for i in ensemble_pred_indices])
        
        # Load test labels
        test_ds = storage.load_split('test', task=task)
        if task == 'clarity':
            label_key = 'clarity_label'
        else:  # evasion
            label_key = 'evasion_label'
        y_test = np.array([test_ds[i][label_key] for i in range(len(test_ds))])
        
        # Compute metrics
        metrics = compute_all_metrics(
            y_test, ensemble_pred, label_list,
            task_name=f"TYPE2_ENSEMBLE_{task}"
        )
        
        print(f"  Ensemble: {len(probabilities_list)} models")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        final_results[task] = {
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba,
            'metrics': metrics,
            'n_models': len(probabilities_list),
            'weights': normalized_weights
        }
        
        # Save predictions and probabilities
        if save_results:
            # Predictions
            pred_path = predictions_dir / f"pred_type2_ensemble_{task}.npy"
            np.save(pred_path, ensemble_pred)
            
            # Probabilities
            proba_path = predictions_dir / f"probs_type2_ensemble_{task}.npy"
            np.save(proba_path, ensemble_proba)
            
            print(f"  ✓ Saved predictions: {pred_path}")
            print(f"  ✓ Saved probabilities: {proba_path}")
    
    # ========================================================================
    # STEP 5: Hierarchical Evaluation (Evasion → Clarity)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: HIERARCHICAL EVALUATION (EVASION → CLARITY)")
    print("="*80)
    
    hierarchical_results = {}
    
    if 'evasion' in final_results and 'clarity' in final_results:
        # Use ensemble evasion predictions
        evasion_pred = final_results['evasion']['predictions']
        
        # Load test split for clarity labels
        test_ds_evasion = storage.load_split('test', task='evasion')
        y_test_clarity = np.array([test_ds_evasion[i]['clarity_label'] for i in range(len(test_ds_evasion))])
        
        # Evaluate hierarchical approach
        hierarchical_metrics = evaluate_hierarchical_approach(
            evasion_predictions=evasion_pred,
            clarity_gold_labels=y_test_clarity,
            clarity_label_list=label_lists['clarity']
        )
        
        hierarchical_results = {
            'metrics': hierarchical_metrics,
            'predictions': hierarchical_metrics.get('predictions', None)
        }
        
        print(f"  Hierarchical Macro F1: {hierarchical_metrics['macro_f1']:.4f}")
        print(f"  Hierarchical Accuracy: {hierarchical_metrics['accuracy']:.4f}")
        
        if save_results:
            if hierarchical_metrics.get('predictions') is not None:
                hier_pred_path = predictions_dir / "pred_type2_hierarchical.npy"
                np.save(hier_pred_path, hierarchical_metrics['predictions'])
                print(f"  ✓ Saved hierarchical predictions: {hier_pred_path}")
    
    # ========================================================================
    # STEP 6: Save results and create visualizations
    # ========================================================================
    if save_results:
        print("\n" + "="*80)
        print("STEP 6: SAVE RESULTS AND METADATA")
        print("="*80)
        
        # Helper function to make JSON serializable
        def make_json_serializable(obj):
            """Recursively convert numpy arrays and types to JSON-serializable Python types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj
        
        # Prepare results for saving
        results_dict = {
            'final_results': {
                task: {
                    'metrics': make_json_serializable(result['metrics']),
                    'n_models': result['n_models'],
                    'weights': make_json_serializable(result['weights'])
                }
                for task, result in final_results.items()
            },
            'hierarchical_results': make_json_serializable(hierarchical_results)
        }
        
        # Save to Drive
        results_path_drive = results_dir / 'final_results_type2.json'
        with open(results_path_drive, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"✓ Saved results to Drive: {results_path_drive}")
        
        # Save to GitHub
        results_path_github = metadata_results_dir / 'final_results_type2.json'
        with open(results_path_github, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"✓ Saved results to GitHub: {results_path_github}")
    
    if create_plots:
        print("\n" + "="*80)
        print("STEP 7: CREATE VISUALIZATIONS")
        print("="*80)
        
        for task, result in final_results.items():
            y_test = None
            test_ds = storage.load_split('test', task=task)
            if task == 'clarity':
                label_key = 'clarity_label'
            else:
                label_key = 'evasion_label'
            y_test = np.array([test_ds[i][label_key] for i in range(len(test_ds))])
            
            y_pred = result['predictions']
            y_proba = result['probabilities']
            label_list = label_lists[task]
            
            # Create plots
            visualize_all_evaluation(
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                label_list=label_list,
                task_name=f"TYPE2_ENSEMBLE_{task}",
                save_dir=str(plots_dir / f"type2_ensemble_{task}")
            )
            
            print(f"  ✓ Created plots for {task}")
    
    print("\n" + "="*80)
    print("✓ FINAL EVALUATION TYPE 2 COMPLETE")
    print("="*80)
    
    return {
        'final_results': final_results,
        'hierarchical_results': hierarchical_results
    }

