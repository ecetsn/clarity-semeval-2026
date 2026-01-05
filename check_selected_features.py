"""
Selected Features Tablosu Oluşturma Scripti
Her model×task×classifier kombinasyonu için seçilmiş feature'ları bulur ve tablo oluşturur.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

def find_selected_features_files(base_path: Path) -> Dict[str, Dict]:
    """
    Tüm selected features dosyalarını bulur ve yükler.
    
    Returns:
        Dict with keys: 'ablation_all', 'ablation_fusion', 'checkpoint_files'
    """
    results = {
        'ablation_all': None,
        'ablation_fusion': None,
        'checkpoint_files': {}
    }
    
    # 1. results/ablation/selected_features_all.json
    ablation_all_path = base_path / 'results' / 'ablation' / 'selected_features_all.json'
    if ablation_all_path.exists():
        try:
            with open(ablation_all_path, 'r') as f:
                results['ablation_all'] = json.load(f)
            print(f"[OK] Found: {ablation_all_path}")
        except Exception as e:
            print(f"[ERROR] Error loading {ablation_all_path}: {e}")
    else:
        print(f"[NOT FOUND] Not found: {ablation_all_path}")
    
    # 2. results/ablation/selected_features_for_early_fusion.json
    ablation_fusion_path = base_path / 'results' / 'ablation' / 'selected_features_for_early_fusion.json'
    if ablation_fusion_path.exists():
        try:
            with open(ablation_fusion_path, 'r') as f:
                results['ablation_fusion'] = json.load(f)
            print(f"[OK] Found: {ablation_fusion_path}")
        except Exception as e:
            print(f"[ERROR] Error loading {ablation_fusion_path}: {e}")
    else:
        print(f"[NOT FOUND] Not found: {ablation_fusion_path}")
    
    # 3. Checkpoint dosyaları: results/FinalResultsType2/classifier_specific/checkpoint/selected_features_{clf}_{task}.json
    checkpoint_dir = base_path / 'results' / 'FinalResultsType2' / 'classifier_specific' / 'checkpoint'
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob('selected_features_*.json'))
        print(f"[OK] Found {len(checkpoint_files)} checkpoint files in {checkpoint_dir}")
        
        for file_path in checkpoint_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # Dosya adından classifier ve task çıkar: selected_features_{clf}_{task}.json
                parts = file_path.stem.replace('selected_features_', '').split('_')
                if len(parts) >= 2:
                    clf_name = parts[0]
                    task = '_'.join(parts[1:])  # task birden fazla kelime olabilir
                    key = f"{clf_name}_{task}"
                    results['checkpoint_files'][key] = {
                        'path': str(file_path),
                        'data': data if isinstance(data, list) else [data] if data else [],
                        'n_features': len(data) if isinstance(data, list) else 0
                    }
            except Exception as e:
                print(f"[ERROR] Error loading {file_path}: {e}")
    else:
        print(f"[NOT FOUND] Checkpoint directory not found: {checkpoint_dir}")
    
    return results

def create_features_table(features_data: Dict) -> pd.DataFrame:
    """
    Selected features verilerinden tablo oluşturur.
    """
    rows = []
    
    # 1. ablation_all.json'dan (format: {model_task_classifier: {...}})
    if features_data['ablation_all']:
        for key, value in features_data['ablation_all'].items():
            # Key format: "model_task_classifier" veya "model_task"
            parts = key.split('_')
            if len(parts) >= 2:
                model = parts[0]
                task = parts[1]
                classifier = parts[2] if len(parts) > 2 else value.get('classifier', 'N/A')
                
                selected_features = value.get('selected_features', [])
                n_features = value.get('n_features', len(selected_features))
                greedy_f1 = value.get('greedy_f1', None)
                
                rows.append({
                    'source': 'ablation_all.json',
                    'model': model,
                    'task': task,
                    'classifier': classifier,
                    'n_features': n_features,
                    'greedy_f1': greedy_f1,
                    'features': selected_features,
                    'file_path': 'results/ablation/selected_features_all.json'
                })
    
    # 2. Checkpoint dosyalarından
    for key, checkpoint_info in features_data['checkpoint_files'].items():
        parts = key.split('_', 1)
        if len(parts) == 2:
            classifier = parts[0]
            task = parts[1]
            
            features = checkpoint_info['data']
            n_features = checkpoint_info['n_features']
            
            rows.append({
                'source': 'checkpoint',
                'model': 'N/A (checkpoint)',
                'task': task,
                'classifier': classifier,
                'n_features': n_features,
                'greedy_f1': None,
                'features': features,
                'file_path': checkpoint_info['path']
            })
    
    # 3. ablation_fusion.json'dan (top-K features, task bazında)
    if features_data['ablation_fusion']:
        for task, task_data in features_data['ablation_fusion'].items():
            if isinstance(task_data, dict):
                features = task_data.get('features', [])
                top_k = task_data.get('top_k', len(features))
                
                rows.append({
                    'source': 'ablation_fusion.json',
                    'model': 'ALL (top-K)',
                    'task': task,
                    'classifier': 'ALL',
                    'n_features': top_k,
                    'greedy_f1': None,
                    'features': features,
                    'file_path': 'results/ablation/selected_features_for_early_fusion.json'
                })
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    return df

def print_summary_table(df: pd.DataFrame):
    """
    Özet tablo yazdırır.
    """
    if df.empty:
        print("\n[WARNING] No selected features data found!")
        return
    
    print("\n" + "="*100)
    print("SELECTED FEATURES SUMMARY TABLE")
    print("="*100)
    
    # Özet: Model × Task × Classifier
    summary_cols = ['source', 'model', 'task', 'classifier', 'n_features', 'greedy_f1', 'file_path']
    if all(col in df.columns for col in summary_cols):
        summary_df = df[summary_cols].copy()
        summary_df = summary_df.sort_values(['model', 'task', 'classifier'])
        
        print("\nSummary (Model × Task × Classifier):")
        print(summary_df.to_string(index=False))
        
        # Pivot table: Model × Task (n_features)
        if 'model' in df.columns and 'task' in df.columns and 'n_features' in df.columns:
            try:
                pivot_df = df.pivot_table(
                    index='model',
                    columns='task',
                    values='n_features',
                    aggfunc='first',
                    fill_value=0
                )
                print("\nPivot Table: Model × Task (Number of Features):")
                print(pivot_df.to_string())
            except Exception as e:
                print(f"\n[WARNING] Could not create pivot table: {e}")
        
        # Classifier × Task pivot
        if 'classifier' in df.columns and 'task' in df.columns and 'n_features' in df.columns:
            try:
                pivot_clf = df.pivot_table(
                    index='classifier',
                    columns='task',
                    values='n_features',
                    aggfunc='first',
                    fill_value=0
                )
                print("\nPivot Table: Classifier × Task (Number of Features):")
                print(pivot_clf.to_string())
            except Exception as e:
                print(f"\n[WARNING] Could not create classifier pivot table: {e}")
    
    # Feature listesi (ilk 5 örnek)
    print("\nSample Feature Lists (first 5 entries):")
    for idx, row in df.head(5).iterrows():
        features = row.get('features', [])
        if features:
            print(f"\n  {row.get('model', 'N/A')} - {row.get('task', 'N/A')} - {row.get('classifier', 'N/A')}:")
            print(f"    Features ({len(features)}): {', '.join(features[:10])}{'...' if len(features) > 10 else ''}")

def main():
    """
    Ana fonksiyon: Selected features dosyalarını bulur ve tablo oluşturur.
    """
    print("="*100)
    print("SELECTED FEATURES LOCATION CHECKER")
    print("="*100)
    
    # Workspace root'u bul
    workspace_root = Path(__file__).parent
    
    print(f"\nSearching in: {workspace_root}")
    print(f"   (Note: If files are in Google Drive, check: /content/drive/MyDrive/semeval_data/)")
    
    # Dosyaları bul
    features_data = find_selected_features_files(workspace_root)
    
    # Tablo oluştur
    df = create_features_table(features_data)
    
    # Özet yazdır
    print_summary_table(df)
    
    # CSV olarak kaydet
    if not df.empty:
        output_path = workspace_root / 'selected_features_table.csv'
        summary_cols = ['source', 'model', 'task', 'classifier', 'n_features', 'greedy_f1', 'file_path']
        if all(col in df.columns for col in summary_cols):
            df[summary_cols].to_csv(output_path, index=False)
            print(f"\n[OK] Saved summary table to: {output_path}")
        
        # Detaylı tablo (features listesi ile)
        detailed_path = workspace_root / 'selected_features_detailed.csv'
        # Features listesini string'e çevir
        df_export = df.copy()
        df_export['features'] = df_export['features'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        df_export.to_csv(detailed_path, index=False)
        print(f"[OK] Saved detailed table to: {detailed_path}")
    
    print("\n" + "="*100)
    print("CHECK COMPLETE")
    print("="*100)
    
    # Özet istatistikler
    if not df.empty:
        print(f"\nStatistics:")
        print(f"   Total entries: {len(df)}")
        print(f"   Unique models: {df['model'].nunique() if 'model' in df.columns else 0}")
        print(f"   Unique tasks: {df['task'].nunique() if 'task' in df.columns else 0}")
        print(f"   Unique classifiers: {df['classifier'].nunique() if 'classifier' in df.columns else 0}")
        print(f"   Average features per entry: {df['n_features'].mean():.1f if 'n_features' in df.columns else 0}")

if __name__ == '__main__':
    main()

