"""
Yarışmaya uygun tek fonksiyon: Tüm final evaluation işlemlerini yapar
"""
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder

from .classifiers import get_classifier_dict
from .inference import predict_batch_from_dataset
from .hierarchical import evaluate_hierarchical_approach
from ..storage.manager import StorageManager
from ..features.extraction import featurize_hf_dataset_in_batches_v2
from ..evaluation.metrics import compute_all_metrics, print_classification_report
from ..evaluation.tables import print_results_table, style_table, style_table_paper
from ..evaluation.visualizer import visualize_all_evaluation


def run_final_evaluation(
    storage: StorageManager,
    models: List[str],
    tasks: List[str],
    model_configs: Dict[str, str],
    model_max_lengths: Dict[str, int],
    label_lists: Dict[str, List[str]],
    device: Optional[torch.device] = None,
    classifiers: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    batch_size: int = 8,
    save_results: bool = True,
    create_plots: bool = True,
) -> Dict[str, Any]:
    """
    YARIŞMAYA UYGUN TEK FONKSİYON: Tüm final evaluation işlemlerini yapar
    
    Bu fonksiyon:
    1. Test feature'larını extract eder (yoksa) veya Drive'dan yükler
    2. Train+Dev üzerinde tüm model×classifier kombinasyonlarını eğitir
    3. Test setinde tahmin yapar ve metrikleri hesaplar
    4. Hierarchical evaluation yapar (evasion → clarity)
    5. Sonuçları kaydeder ve görselleştirir
    
    Args:
        storage: StorageManager instance
        models: Model listesi (e.g., ['bert', 'roberta', 'deberta', 'xlnet'])
        tasks: Task listesi (e.g., ['clarity', 'evasion'])
        model_configs: Model isimleri mapping (e.g., {'bert': 'bert-base-uncased'})
        model_max_lengths: Max sequence length mapping (e.g., {'bert': 512})
        label_lists: Label listeleri (e.g., {'clarity': ['Ambivalent', ...], 'evasion': [...]})
        device: torch device (None ise otomatik seçilir)
        classifiers: Classifier dict (None ise default 6 classifier)
        random_state: Random seed
        batch_size: Feature extraction batch size
        save_results: Sonuçları kaydet (True/False)
        create_plots: Plot'ları oluştur (True/False)
    
    Returns:
        Dictionary: {
            'final_results': {
                model_name: {
                    task_name: {
                        classifier_name: {
                            'predictions': np.ndarray,
                            'probabilities': np.ndarray,
                            'metrics': dict
                        }
                    }
                }
            },
            'hierarchical_results': {
                model_name: {
                    'metrics': dict,
                    'classifier': str
                }
            }
        }
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if classifiers is None:
        classifiers = get_classifier_dict(random_state=random_state)
    
    # ========================================================================
    # CRITICAL: Create ALL output directories upfront to prevent FileNotFoundError
    # This ensures all directories exist before any save operations
    # Only for FinalResultsType1 (5. notebook), not affecting other notebooks
    # Always create directories (exist_ok=True makes it safe)
    # ========================================================================
    # Drive directories (for large files: plots, predictions, tables, results)
    plots_dir = storage.data_path / 'results/FinalResultsType1/plots'
    predictions_dir = storage.data_path / 'results/FinalResultsType1/predictions'
    tables_dir = storage.data_path / 'results/FinalResultsType1/tables'
    results_dir = storage.data_path / 'results/FinalResultsType1'
    
    # GitHub directories (for metadata: JSON results)
    metadata_results_dir = storage.github_path / 'results/FinalResultsType1Results'
    
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
    
    print("="*80)
    print("FINAL EVALUATION: YARIŞMAYA UYGUN TEK FONKSİYON")
    print("="*80)
    print(f"Models: {models}")
    print(f"Tasks: {tasks}")
    print(f"Classifiers: {list(classifiers.keys())}")
    print(f"Device: {device}")
    print(f"Checkpoint: Always enabled (load if exists, extract and save if not)")
    print("="*80)
    
    final_results = {}
    hierarchical_results = {}
    
    # ========================================================================
    # ADIM 1: Test feature'larını extract et veya yükle
    # ========================================================================
    print("\n" + "="*80)
    print("ADIM 1: TEST FEATURE EXTRACTION")
    print("="*80)
    
    for model_key in models:
        print(f"\n{'-'*60}")
        print(f"Model: {model_key.upper()}")
        print(f"{'-'*60}")
        
        # Load transformer model and tokenizer
        model_name = model_configs[model_key]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model = AutoModel.from_pretrained(model_name)
        hf_model.to(device)
        hf_model.eval()
        
        # Get model-specific max sequence length
        # Use provided model_max_lengths if valid, otherwise simple fallback
        if model_key in model_max_lengths and model_max_lengths[model_key] > 0:
            max_seq_len = model_max_lengths[model_key]
        else:
            # Simple fallback: XLNet uses 1024, others use 512
            max_seq_len = 1024 if 'xlnet' in model_name.lower() else 512
            print(f"  Warning: Using fallback max_seq_len={max_seq_len} for {model_key}")
        
        print(f"  Max sequence length: {max_seq_len}")
        
        for task in tasks:
            print(f"\n  Task: {task}")
            
            # ZORUNLU: Test feature'ları mutlaka olmalı (ya Drive'da var ya da extract edilmeli)
            # Checkpoint mantığı: Önce Drive'dan yükle, yoksa extract et ve kaydet
            try:
                X_test = storage.load_features(model_key, task, 'test')
                print(f"    ✓ Loaded from Drive: {X_test.shape}")
            except FileNotFoundError:
                # Test feature'ları Drive'da yok → MUTLAKA extract et ve kaydet
                print(f"    → Test features not found in Drive. Extracting and saving...")
                
                # Load splits
                test_ds = storage.load_split('test', task=task)
                train_ds = storage.load_split('train', task=task)
                
                # Fit TF-IDF on train (required for test feature extraction)
                print(f"      Fitting TF-IDF on train split...")
                _, _, tfidf_vectorizer = featurize_hf_dataset_in_batches_v2(
                    train_ds, tokenizer, hf_model, device,
                    question_key='interview_question',
                    answer_key='interview_answer',
                    batch_size=batch_size,
                    max_sequence_length=max_seq_len,
                    show_progress=False,
                    tfidf_vectorizer=None
                )
                
                # Extract test features
                print(f"      Extracting test features...")
                X_test, feature_names, _ = featurize_hf_dataset_in_batches_v2(
                    test_ds, tokenizer, hf_model, device,
                    question_key='interview_question',
                    answer_key='interview_answer',
                    batch_size=batch_size,
                    max_sequence_length=max_seq_len,
                    show_progress=True,
                    tfidf_vectorizer=tfidf_vectorizer
                )
                
                # ZORUNLU: Extract edilen feature'ları MUTLAKA Drive'a kaydet (checkpoint)
                storage.save_features(X_test, model_key, task, 'test', feature_names)
                print(f"    ✓ Extracted and saved to Drive: {X_test.shape}")
                print(f"      Next time, features will be loaded from Drive (no re-extraction needed)")
        
        # Free GPU memory
        del hf_model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ========================================================================
    # ADIM 2: Train+Dev üzerinde eğit, Test setinde değerlendir
    # ========================================================================
    print("\n" + "="*80)
    print("ADIM 2: FINAL TRAINING & TEST EVALUATION")
    print("="*80)
    
    for model_key in models:
        print(f"\n{'-'*80}")
        print(f"MODEL: {model_key.upper()}")
        print(f"{'-'*80}")
        
        final_results[model_key] = {}
        
        for task in tasks:
            print(f"\n  Task: {task.upper()}")
            
            label_list = label_lists[task]
            label_key = 'clarity_label' if task == 'clarity' else 'evasion_label'
            
            # Load splits
            test_ds = storage.load_split('test', task=task)
            train_ds = storage.load_split('train', task=task)
            dev_ds = storage.load_split('dev', task=task)
            
            # Load features
            X_train = storage.load_features(model_key, task, 'train')
            X_dev = storage.load_features(model_key, task, 'dev')
            X_test = storage.load_features(model_key, task, 'test')
            
            # Extract labels
            y_train = np.array([train_ds[i][label_key] for i in range(len(train_ds))])
            y_dev = np.array([dev_ds[i][label_key] for i in range(len(dev_ds))])
            y_test = np.array([test_ds[i][label_key] for i in range(len(test_ds))])
            
            # Combine train+dev
            X_train_full = np.vstack([X_train, X_dev])
            y_train_full = np.concatenate([y_train, y_dev])
            
            print(f"    Training: {X_train_full.shape[0]} samples (train+dev)")
            print(f"    Testing: {X_test.shape[0]} samples")
            
            # Train and evaluate
            task_results = {}
            
            for clf_name, clf in classifiers.items():
                print(f"\n    Training {clf_name}...")
                
                # Train on train+dev
                clf.fit(X_train_full, y_train_full)
                
                # Predict on test
                y_test_pred = clf.predict(X_test)
                
                try:
                    y_test_proba = clf.predict_proba(X_test)
                except AttributeError:
                    y_test_proba = None
                
                # Compute metrics
                metrics = compute_all_metrics(
                    y_test, y_test_pred, label_list,
                    task_name=f"TEST_{model_key}_{task}_{clf_name}"
                )
                
                # 1. Print classification report first
                print_classification_report(
                    y_test, y_test_pred, label_list,
                    task_name=f"TEST - {model_key} - {task} - {clf_name}"
                )
                
                # 2. Show confusion matrix immediately after report
                if create_plots:
                    from ..evaluation.plots import plot_confusion_matrix
                    # Directory already created at function start
                    plots_dir = storage.data_path / 'results/FinalResultsType1/plots'
                    plot_confusion_matrix(
                        y_test, y_test_pred, label_list,
                        task_name=f"TEST - {model_key} - {task} - {clf_name}",
                        save_path=str(plots_dir / f'confusion_matrix_{clf_name}_TEST_{model_key}_{task}.png')
                    )
                
                # 3. Create other plots (PR/ROC) - save only, don't display (to avoid clutter)
                if create_plots and y_test_proba is not None:
                    from ..evaluation.plots import plot_precision_recall_curves, plot_roc_curves
                    # Directory already created at function start
                    plots_dir = storage.data_path / 'results/FinalResultsType1/plots'
                    plot_precision_recall_curves(
                        y_test, y_test_proba, label_list,
                        task_name=f"TEST - {model_key} - {task} - {clf_name}",
                        save_path=str(plots_dir / f'precision_recall_{clf_name}_TEST_{model_key}_{task}.png')
                    )
                    plot_roc_curves(
                        y_test, y_test_proba, label_list,
                        task_name=f"TEST - {model_key} - {task} - {clf_name}",
                        save_path=str(plots_dir / f'roc_{clf_name}_TEST_{model_key}_{task}.png')
                    )
                
                task_results[clf_name] = {
                    'predictions': y_test_pred,
                    'probabilities': y_test_proba,
                    'metrics': metrics
                }
                
                # Save predictions (NO probabilities - not needed for final evaluation)
                if save_results:
                    storage.save_predictions(
                        y_test_pred, model_key, clf_name, task, 'test',
                        save_dir=str(storage.data_path / 'results/FinalResultsType1/predictions'),
                        metadata_dir='results/FinalResultsType1Results'
                    )
            
            # Print comparison table
            print_results_table(
                {name: {'metrics': res['metrics']} for name, res in task_results.items()},
                task_name=f"TEST - {model_key} - {task}",
                sort_by="Macro F1"
            )
            
            final_results[model_key][task] = task_results
            
            # Save results metadata
            if save_results:
                experiment_id = f"FINAL_TEST_{model_key}_{task}"
                storage.save_results({
                    'split': 'test',
                    'model': model_key,
                    'task': task,
                    'n_test': len(y_test),
                    'results': {
                        name: {'metrics': res['metrics']}
                        for name, res in task_results.items()
                    }
                }, experiment_id, save_dir='results/FinalResultsType1Results')
    
    # ========================================================================
    # ADIM 3: Hierarchical Evaluation (Evasion → Clarity)
    # ========================================================================
    print("\n" + "="*80)
    print("ADIM 3: HIERARCHICAL EVALUATION (EVASION → CLARITY)")
    print("="*80)
    
    for model_key in models:
        if 'evasion' not in final_results.get(model_key, {}):
            continue
        
        print(f"\n{'-'*80}")
        print(f"MODEL: {model_key.upper()}")
        print(f"{'-'*80}")
        
        evasion_results = final_results[model_key]['evasion']
        
        # Find best classifier by Macro F1
        best_classifier = None
        best_f1 = -1
        
        for clf_name, clf_result in evasion_results.items():
            f1 = clf_result['metrics'].get('macro_f1', 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_classifier = clf_name
        
        if best_classifier is None:
            continue
        
        print(f"  Using evasion predictions from: {best_classifier} (F1: {best_f1:.4f})")
        
        # Load test set
        test_ds_evasion = storage.load_split('test', task='evasion')
        
        # Get predictions and true labels
        y_evasion_pred_test = evasion_results[best_classifier]['predictions']
        y_evasion_true_test = np.array([test_ds_evasion[i]['evasion_label'] for i in range(len(test_ds_evasion))])
        y_clarity_true_test = np.array([test_ds_evasion[i]['clarity_label'] for i in range(len(test_ds_evasion))])
        
        # Encode labels
        le_evasion = LabelEncoder()
        le_clarity = LabelEncoder()
        
        y_evasion_true_encoded = le_evasion.fit_transform(y_evasion_true_test)
        y_clarity_true_encoded = le_clarity.fit_transform(y_clarity_true_test)
        y_evasion_pred_encoded = le_evasion.transform(y_evasion_pred_test)
        
        # Evaluate hierarchical approach
        hierarchical_metrics = evaluate_hierarchical_approach(
            y_evasion_true_encoded,
            y_evasion_pred_encoded,
            y_clarity_true_encoded,
            label_lists['evasion'],
            label_lists['clarity']
        )
        
        hierarchical_results[model_key] = {
            'classifier': best_classifier,
            'metrics': hierarchical_metrics,
            'evasion_f1': best_f1
        }
        
        print(f"\n  Hierarchical Clarity Performance:")
        print(f"    Accuracy: {hierarchical_metrics['accuracy']:.4f}")
        print(f"    Macro F1: {hierarchical_metrics['macro_f1']:.4f}")
        print(f"    Weighted F1: {hierarchical_metrics['weighted_f1']:.4f}")
        
        print_classification_report(
            y_clarity_true_encoded,
            hierarchical_metrics['predictions'],
            label_lists['clarity'],
            task_name=f"TEST - {model_key} - Hierarchical Evasion→Clarity"
        )
        
        # Save hierarchical predictions
        if save_results:
            storage.save_predictions(
                hierarchical_metrics['predictions'],
                model_key, best_classifier, 'hierarchical_evasion_to_clarity', 'test',
                save_dir=str(storage.data_path / 'results/FinalResultsType1/predictions'),
                metadata_dir='results/FinalResultsType1Results'
            )
            
            experiment_id = f"FINAL_TEST_{model_key}_hierarchical_evasion_to_clarity"
            storage.save_results({
                'split': 'test',
                'model': model_key,
                'task': 'hierarchical_evasion_to_clarity',
                'n_test': len(y_clarity_true_test),
                'evasion_classifier': best_classifier,
                'evasion_f1': best_f1,
                'results': {
                    best_classifier: {'metrics': hierarchical_metrics}
                }
            }, experiment_id, save_dir='results/FinalResultsType1Results')
    
    # ========================================================================
    # ADIM 4: Final Summary Tables (Model-wise and Classifier-wise)
    # ========================================================================
    if save_results:
        print("\n" + "="*80)
        print("ADIM 4: FINAL SUMMARY TABLES GENERATION")
        print("="*80)
        
        # Collect all results for summary
        summary_rows = []
        all_tasks = tasks + ['hierarchical_evasion_to_clarity']
        
        for model_key in models:
            if model_key not in final_results:
                continue
            for task in all_tasks:
                if task not in final_results[model_key]:
                    continue
                task_results = final_results[model_key][task]
                for clf_name, result in task_results.items():
                    if 'metrics' in result:
                        metrics = result['metrics']
                        summary_rows.append({
                            'model': model_key,
                            'classifier': clf_name,
                            'task': task,
                            'macro_f1': metrics.get('macro_f1', 0.0),
                            'weighted_f1': metrics.get('weighted_f1', 0.0),
                            'accuracy': metrics.get('accuracy', 0.0)
                        })
        
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            
            # MODEL-WISE TABLES: For each model, Classifier × Tasks
            print("\n" + "="*100)
            print("FINAL SUMMARY — MODEL-WISE (Classifier × Tasks)")
            print("="*100)
            
            for model_name in sorted(df_summary['model'].unique()):
                print(f"\nMODEL: {model_name.upper()}")
                
                df_model = df_summary[df_summary['model'] == model_name]
                
                # Pivot: Classifier × Tasks
                pivot_model = df_model.pivot_table(
                    index='classifier',
                    columns='task',
                    values='macro_f1'
                )
                
                if not pivot_model.empty:
                    # Print unformatted table
                    print(pivot_model.to_string())
                    
                    # Style and save table (paper-ready: bold+green for best, italic for hierarchical>clarity)
                    styled_model = style_table_paper(pivot_model, precision=4, apply_column_mapping=False)
                    storage.save_table(
                        styled_model,
                        table_name=f'final_test_model_wise_{model_name}',
                        formats=['csv', 'html', 'png', 'tex', 'md'],
                        save_dir=str(storage.data_path / 'results/FinalResultsType1/tables'),
                        use_paper_style=True
                    )
            
            # CLASSIFIER-WISE TABLES: For each classifier, Model × Tasks
            print("\n" + "="*100)
            print("FINAL SUMMARY — CLASSIFIER-WISE (Model × Tasks)")
            print("="*100)
            
            for clf_name in sorted(df_summary['classifier'].unique()):
                print(f"\nCLASSIFIER: {clf_name.upper()}")
                
                df_clf = df_summary[df_summary['classifier'] == clf_name]
                
                # Pivot: Model × Tasks
                pivot_clf = df_clf.pivot_table(
                    index='model',
                    columns='task',
                    values='macro_f1'
                )
                
                if not pivot_clf.empty:
                    # Print unformatted table
                    print(pivot_clf.to_string())
                    
                    # Style and save table (paper-ready: bold+green for best, italic for hierarchical>clarity)
                    styled_clf = style_table_paper(pivot_clf, precision=4, apply_column_mapping=False)
                    storage.save_table(
                        styled_clf,
                        table_name=f'final_test_classifier_wise_{clf_name}',
                        formats=['csv', 'html', 'png', 'tex', 'md'],
                        save_dir=str(storage.data_path / 'results/FinalResultsType1/tables'),
                        use_paper_style=True
                    )
            
            # Save all results dictionary
            storage.save_all_results_dict(
                {'final_results': final_results, 'hierarchical_results': hierarchical_results},
                filename='all_results_test.pkl',
                save_dir='results/FinalResultsType1'
            )
        else:
            print("WARNING: No results available for summary tables")
    
    print("\n" + "="*80)
    print("FINAL EVALUATION COMPLETE")
    print("="*80)
    
    return {
        'final_results': final_results,
        'hierarchical_results': hierarchical_results
    }

