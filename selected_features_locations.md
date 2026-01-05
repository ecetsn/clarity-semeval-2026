# Selected Features Lokasyon Tablosu

Bu dokümantasyon, her model×task×classifier kombinasyonu için seçilmiş feature'ların nerede saklandığını gösterir.

## 📍 Dosya Lokasyonları

### 1. **Ablation Study Sonuçları (Greedy Selection)**

**Dosya:** `results/ablation/selected_features_all.json`

**Format:**
```json
{
  "model_task_classifier": {
    "model": "bert",
    "task": "clarity",
    "classifier": "LogisticRegression",
    "selected_features": ["feature1", "feature2", ...],
    "n_features": 20,
    "greedy_f1": 0.85
  }
}
```

**Key Format:** `{model}_{task}_{classifier}`

**Örnek Keys:**
- `bert_clarity_LogisticRegression`
- `roberta_evasion_XGBoost`
- `deberta_clarity_MLP`

**Not:** Bu dosya `03_5_ablation_study.ipynb` veya `notebooks/ablation.py` çalıştırıldığında oluşturulur.

---

### 2. **Top-K Features for Early Fusion**

**Dosya:** `results/ablation/selected_features_for_early_fusion.json`

**Format:**
```json
{
  "clarity": {
    "top_k": 10,
    "features": ["feature1", "feature2", ...],
    "ranking": [...]
  },
  "evasion": {
    "top_k": 10,
    "features": ["feature1", "feature2", ...],
    "ranking": [...]
  }
}
```

**Kullanım:** Tüm modeller için aynı top-K feature'lar (task bazında)

**Not:** Bu dosya ablation study'de global top-K feature seçimi yapıldığında oluşturulur.

---

### 3. **Checkpoint Dosyaları (60-Feature System)**

**Dizin:** `results/FinalResultsType2/classifier_specific/checkpoint/`

**Dosya Format:** `selected_features_{classifier}_{task}.json`

**Örnek Dosyalar:**
- `selected_features_LogisticRegression_clarity.json`
- `selected_features_XGBoost_evasion.json`
- `selected_features_MLP_clarity.json`

**Format:**
```json
["feature1", "feature2", "feature3", ...]
```

**Not:** Bu dosyalar `notebooks/ablation.py` veya `03_5_ablation_study.ipynb` çalıştırıldığında checkpoint olarak kaydedilir.

**İlgili Checkpoint Dosyaları:**
- `selected_features_{clf}_{task}.json` - Seçilmiş feature listesi
- `trajectory_{clf}_{task}.csv` - Greedy selection trajectory (n_features, macro_f1)
- `greedy_checkpoint_{clf}_{task}.pkl` - Greedy selection intermediate state
- `metrics_{clf}_{task}.json` - Test set metrics
- `{clf}_{task}_predictions.npy` - Test predictions
- `{clf}_{task}_probabilities.npy` - Test probabilities

---

## 📊 Model × Task × Classifier Kombinasyonları

### Modeller
- `bert`
- `bert_political`
- `bert_ambiguity`
- `roberta`
- `deberta`
- `xlnet`

### Task'lar
- `clarity` (3 classes)
- `evasion` (9 classes)

### Classifier'lar
- `LogisticRegression`
- `LinearSVC`
- `RandomForest`
- `MLP`
- `XGBoost`
- `LightGBM`

### Toplam Kombinasyon Sayısı
- **6 modeller** × **2 task** × **6 classifier** = **72 kombinasyon**

---

## 🔍 Dosyaları Bulma

### Google Drive'da (Colab)
```python
# StorageManager kullanarak
storage = StorageManager(
    base_path='/content/semeval-context-tree-modular',
    data_path='/content/drive/MyDrive/semeval_data'
)

# Ablation dosyası
ablation_dir = storage.data_path / 'results/ablation'
selected_features_path = ablation_dir / 'selected_features_all.json'

# Checkpoint dosyaları
checkpoint_dir = storage.data_path / 'results/FinalResultsType2/classifier_specific/checkpoint'
```

### Lokal Workspace'te
```python
from pathlib import Path

workspace_root = Path(__file__).parent

# Ablation dosyası
ablation_dir = workspace_root / 'results' / 'ablation'
selected_features_path = ablation_dir / 'selected_features_all.json'

# Checkpoint dosyaları
checkpoint_dir = workspace_root / 'results' / 'FinalResultsType2' / 'classifier_specific' / 'checkpoint'
```

---

## 📋 Örnek Tablo: Her Kombinasyon için Feature Sayısı

| Model | Task | Classifier | n_features | Source | File Path |
|-------|------|------------|------------|--------|-----------|
| bert | clarity | LogisticRegression | 20 | ablation_all.json | results/ablation/selected_features_all.json |
| bert | clarity | XGBoost | 20 | ablation_all.json | results/ablation/selected_features_all.json |
| roberta | evasion | MLP | 20 | ablation_all.json | results/ablation/selected_features_all.json |
| ... | ... | ... | ... | ... | ... |
| ALL | clarity | ALL | 10 | ablation_fusion.json | results/ablation/selected_features_for_early_fusion.json |
| ALL | evasion | ALL | 10 | ablation_fusion.json | results/ablation/selected_features_for_early_fusion.json |
| N/A | clarity | LogisticRegression | 40 | checkpoint | results/FinalResultsType2/.../selected_features_LogisticRegression_clarity.json |
| ... | ... | ... | ... | ... | ... |

---

## 🚀 Dosyaları Yükleme

### Python ile Yükleme

```python
import json
from pathlib import Path

# 1. Ablation all.json
ablation_path = Path('results/ablation/selected_features_all.json')
if ablation_path.exists():
    with open(ablation_path, 'r') as f:
        selected_features_all = json.load(f)
    
    # Her kombinasyon için
    for key, value in selected_features_all.items():
        model = value['model']
        task = value['task']
        classifier = value['classifier']
        features = value['selected_features']
        n_features = value['n_features']
        print(f"{model}_{task}_{classifier}: {n_features} features")

# 2. Checkpoint dosyaları
checkpoint_dir = Path('results/FinalResultsType2/classifier_specific/checkpoint')
for file_path in checkpoint_dir.glob('selected_features_*.json'):
    with open(file_path, 'r') as f:
        features = json.load(f)
    # Dosya adından classifier ve task çıkar
    parts = file_path.stem.replace('selected_features_', '').split('_')
    classifier = parts[0]
    task = '_'.join(parts[1:])
    print(f"{classifier}_{task}: {len(features)} features")
```

### StorageManager ile Yükleme

```python
from src.storage.manager import StorageManager
from src.models.ensemble import load_greedy_selected_features

storage = StorageManager(...)

# Greedy selected features yükle (model bazında)
greedy_features = load_greedy_selected_features(storage, task='clarity')
# Returns: {'bert': [...], 'roberta': [...], ...}
```

---

## ⚠️ Notlar

1. **Cache Kontrolü:** Dosyalar Google Drive'da saklanıyorsa, lokal workspace'te görünmeyebilir. Colab'da çalıştırırken `storage.data_path` kullanın.

2. **Checkpoint Dosyaları:** `notebooks/ablation.py` checkpoint mekanizması kullanır. Eğer bir kombinasyon için checkpoint varsa, greedy selection tekrar çalıştırılmaz.

3. **Feature Sayıları:**
   - **Ablation (25 features):** Her model için 25 feature (7 model-dependent + 18 model-independent), greedy ile 20'ye düşürülür
   - **Early Fusion (60 features):** 60 feature (18 model-independent + 42 model-dependent), global top-20 + greedy 20 = 40 feature

4. **Dosya Formatları:**
   - `selected_features_all.json`: Dict format (key: model_task_classifier)
   - `selected_features_for_early_fusion.json`: Dict format (key: task)
   - Checkpoint dosyaları: List format (sadece feature isimleri)

---

## 📝 Tablo Oluşturma Scripti

`check_selected_features.py` scripti bu dosyaları bulur ve tablo oluşturur:

```bash
python check_selected_features.py
```

**Çıktılar:**
- `selected_features_table.csv` - Özet tablo
- `selected_features_detailed.csv` - Detaylı tablo (feature listeleri ile)

**Not:** Script lokal workspace'te çalışır. Google Drive'daki dosyalar için Colab'da çalıştırın veya `storage.data_path` kullanın.

