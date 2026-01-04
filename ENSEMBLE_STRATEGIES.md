# Ensemble ve Early Fusion Stratejileri

Bu dokümantasyon, greedy selection sonuçlarıyla early fusion ve ensemble yapma stratejilerini açıklar.

## 📊 Mevcut Durum

### 1. **Greedy Selection Sonuçları**
- **Konum**: `results/ablation/selected_features_all.json`
- **Format**: `{model_task: {model, task, classifier, selected_features, n_features}}`
- **Örnek**: `{"bert_clarity": {"model": "bert", "task": "clarity", "selected_features": [...]}}`

### 2. **Top-K Feature Selection**
- **Konum**: `results/ablation/selected_features_for_early_fusion.json`
- **Format**: Task bazında top-10 feature listesi
- **Kullanım**: Tüm modeller için aynı top-K feature'lar

### 3. **Final Evaluation Sonuçları**
- **Format**: `{model: {task: {classifier: {predictions, probabilities, metrics}}}}`
- **Metric**: `macro_f1` (imbalanced class için ideal)

---

## 🎯 Önerilen Stratejiler

### **Strateji 1: Greedy-Based Early Fusion (Model-Specific)**

Her model için greedy ile seçilmiş feature'ları kullanarak early fusion yapın.

```python
from src.models.ensemble import create_greedy_fused_features
from src.storage.manager import StorageManager

# 1. Greedy-selected features'ı yükle (otomatik)
X_train_fused, feature_names = create_greedy_fused_features(
    storage=storage,
    models=['bert', 'roberta', 'deberta', 'xlnet'],
    task='clarity',
    split='train',
    auto_load_greedy=True  # Otomatik olarak greedy features yükler
)

# 2. Classifier'ları eğit
from src.models.classifiers import train_classifiers
results = train_classifiers(
    X_train_fused, y_train,
    X_dev_fused, y_dev,
    classifiers=classifiers
)
```

**Avantajlar:**
- ✅ Her model için en iyi feature'ları kullanır
- ✅ Model-specific optimizasyon
- ✅ Daha az feature = daha hızlı training

**Dezavantajlar:**
- ⚠️ Her model için farklı feature set'i (karşılaştırma zor)

---

### **Strateji 2: Top-K Model Selection + Top-K Features**

En iyi K model'i seç, her birinde top-K feature kullan.

```python
from src.models.ensemble import (
    select_top_models_by_f1,
    create_topk_fused_features
)

# 1. Final evaluation sonuçlarından top-10 model seç
top_models = select_top_models_by_f1(
    results=final_results,  # run_final_evaluation'dan gelen sonuçlar
    task='clarity',
    top_k=10,
    metric='macro_f1'
)

# 2. Top-10 model'ler için top-10 feature'larla early fusion
X_train_fused, feature_names = create_topk_fused_features(
    storage=storage,
    top_models=top_models,
    task='clarity',
    top_k_features=10,  # Her model için top-10 feature
    split='train'
)
```

**Avantajlar:**
- ✅ En iyi performans gösteren modelleri kullanır
- ✅ Tutarlı feature set (tüm modeller için aynı top-K)
- ✅ Top-5 ve Top-10 varyasyonları kolay

---

### **Strateji 3: Late Fusion (Ensemble) - Önerilen**

Birden fazla modelin prediction'larını birleştirin.

```python
from src.models.ensemble import ensemble_from_results

# 1. Top-10 model'lerden ensemble oluştur
ensemble_result = ensemble_from_results(
    results=final_results,
    task='clarity',
    top_k=10,  # Top-10 model
    ensemble_method='weighted_mean',  # F1 score'a göre ağırlıklandırılmış
    metric='macro_f1'
)

# 2. Ensemble predictions ve probabilities
y_ensemble_pred = ensemble_result['predictions']
y_ensemble_proba = ensemble_result['probabilities']

# 3. Evaluate
from src.evaluation.metrics import compute_all_metrics
metrics = compute_all_metrics(
    y_true, y_ensemble_pred, label_list,
    task_name="ENSEMBLE_TOP10"
)
```

**Ensemble Metodları:**
- `'hard_voting'`: Majority vote (sadece predictions)
- `'mean'`: Probability'leri ortalama
- `'weighted_mean'`: F1 score'a göre ağırlıklandırılmış ortalama ⭐ **Önerilen**
- `'max'`: Max pooling

**Avantajlar:**
- ✅ En basit ve etkili yöntem
- ✅ Model diversity'den faydalanır
- ✅ Imbalanced class için weighted_mean ideal

---

### **Strateji 4: Hybrid (Early Fusion + Ensemble)**

Hem early fusion hem de ensemble kullanın.

```python
# 1. Greedy-based early fusion ile birkaç model grubu oluştur
# Grup 1: Top-5 model, top-5 features
top5_models = select_top_models_by_f1(results, task='clarity', top_k=5)
X_fused_top5 = create_topk_fused_features(storage, top5_models, task='clarity', top_k_features=5)

# Grup 2: Top-10 model, top-10 features
top10_models = select_top_models_by_f1(results, task='clarity', top_k=10)
X_fused_top10 = create_topk_fused_features(storage, top10_models, task='clarity', top_k_features=10)

# 2. Her grup için classifier eğit
results_top5 = train_classifiers(X_fused_top5, y_train, X_fused_top5_dev, y_dev)
results_top10 = train_classifiers(X_fused_top10, y_train, X_fused_top10_dev, y_dev)

# 3. Grupları ensemble et
ensemble_predictions = ensemble_predictions_voting([
    results_top5['best_classifier']['dev_pred'],
    results_top10['best_classifier']['dev_pred']
])
```

---

## 🔬 Hangi Modelleri Kullanmalı?

### **Paper'daki Modeller:**
- ✅ **BERT** (bert-base-uncased)
- ✅ **RoBERTa** (roberta-base)
- ✅ **DeBERTa** (microsoft/deberta-base)
- ✅ **XLNet** (xlnet-base-cased)

### **Ek Modeller (Paper'da Yok):**
- ✅ **BERT-Political**: Political domain'e fine-tune edilmiş
- ✅ **BERT-Ambiguity**: Ambiguity detection için fine-tune edilmiş

### **Öneri:**
1. **Temel Ensemble**: Paper'daki 4 model (BERT, RoBERTa, DeBERTa, XLNet)
2. **Extended Ensemble**: + BERT-Political, BERT-Ambiguity (6 model)
3. **Selective Ensemble**: Top-5 veya Top-10 by macro F1

---

## 📈 Imbalanced Class için Öneriler

### **1. Macro F1 Kullanın**
```python
# Doğru metric
metric = 'macro_f1'  # Her class'a eşit ağırlık

# Yanlış metric
metric = 'weighted_f1'  # Class frequency'ye göre ağırlık (majority class'ı önceliklendirir)
```

### **2. Weighted Ensemble**
```python
# F1 score'a göre ağırlıklandırılmış ensemble
ensemble_method = 'weighted_mean'  # Yüksek F1'li modeller daha fazla ağırlık alır
```

### **3. Class-Balanced Sampling (Opsiyonel)**
Eğer training sırasında kullanmak isterseniz:
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE + Under-sampling kombinasyonu
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## 🚀 Hızlı Başlangıç

### **En Basit Ensemble (Önerilen):**
```python
# 1. Final evaluation çalıştır
results = run_final_evaluation(
    storage=storage,
    models=['bert', 'roberta', 'deberta', 'xlnet', 'bert_political', 'bert_ambiguity'],
    tasks=['clarity', 'evasion'],
    ...
)

# 2. Top-10 ensemble
from src.models.ensemble import ensemble_from_results

for task in ['clarity', 'evasion']:
    ensemble = ensemble_from_results(
        results=results,
        task=task,
        top_k=10,
        ensemble_method='weighted_mean'
    )
    
    # 3. Evaluate
    metrics = compute_all_metrics(
        y_test, ensemble['predictions'], label_list,
        task_name=f"ENSEMBLE_TOP10_{task}"
    )
    
    print(f"{task} - Ensemble Macro F1: {metrics['macro_f1']:.4f}")
```

---

## 📝 İhsan ve Ece'nin Reposu

**Not**: İhsan ve Ece'nin reposu bu workspace'te bulunamadı. Ancak genel öneriler:

### **Paper'daki Yaklaşımlar:**
1. **Evasion-based Clarity**: İki aşamalı classification (evasion → clarity mapping)
2. **Hierarchical Taxonomy**: Fine-grained labels kullanarak high-level prediction
3. **LoRA Fine-tuning**: Llama-70b gibi LLM'ler için

### **Sizin Yaklaşımınız:**
- ✅ Context Tree features (paper'da yok) - **YENİLİK**
- ✅ Greedy feature selection (paper'da yok) - **YENİLİK**
- ✅ Multiple transformer models ensemble (paper'da var ama farklı)

### **Öneri:**
Paper'daki modelleri kullanın ama **sizin feature extraction ve selection metodunuzla**:
- Paper: LLM-based (Llama, Falcon, ChatGPT)
- Siz: Transformer features + Context Tree + Greedy Selection ⭐

Bu kombinasyon **hem paper'daki yaklaşımdan farklı hem de potansiyel olarak daha iyi** olabilir.

---

## 🎯 Sonuç ve Öneriler

### **En İyi Strateji (Imbalanced Class için):**
1. ✅ **Late Fusion (Ensemble)** - `weighted_mean` method
2. ✅ **Top-10 models** by macro F1
3. ✅ **Macro F1** metric kullan

### **Alternatif Strateji:**
1. ✅ **Greedy-based Early Fusion** - Model-specific features
2. ✅ **Top-5 models** - Daha az model, daha hızlı

### **Kod Örnekleri:**
Tüm kod örnekleri `src/models/ensemble.py` modülünde mevcut.

