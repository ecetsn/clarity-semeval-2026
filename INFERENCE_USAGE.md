# Inference Fonksiyonu Kullanım Kılavuzu

## `predict_batch_from_dataset()` Fonksiyonu

### Ne Yapar?

1. **Drive'dan Test Feature'larını Kontrol Eder:**
   - `features/raw/X_test_{model_name}_{task}.npy` dosyasını arar
   - Eğer varsa: Drive'dan yükler (tekrar extract etmez)
   - Eğer yoksa: Feature extraction yapar ve Drive'a kaydeder

2. **Prediction Yapar:**
   - Extract edilmiş (veya yüklenmiş) feature'lar üzerinde
   - Trained classifier ile prediction yapar

### Gerekli Argümanlar

```python
predictions = predict_batch_from_dataset(
    # ZORUNLU ARGÜMANLAR:
    dataset=test_ds,                    # Test dataset (list of dicts)
    model=model,                        # HuggingFace transformer model
    tokenizer=tokenizer,                # HuggingFace tokenizer
    classifier=trained_classifier,      # Trained sklearn classifier
    device=device,                      # torch.device('cuda' or 'cpu')
    
    # CHECKPOINT İÇİN ZORUNLU (Drive'dan yükleme için):
    storage_manager=storage,            # StorageManager instance
    model_name='bert',                  # Model name (e.g., 'bert', 'roberta')
    task='clarity',                     # Task name ('clarity' or 'evasion')
    
    # TF-IDF İÇİN ZORUNLU:
    tfidf_vectorizer=tfidf_vectorizer,  # Train set'ten fit edilmiş TF-IDF
    
    # OPSİYONEL ARGÜMANLAR:
    question_key='interview_question',  # Dataset'teki question key (default: 'interview_question')
    answer_key='interview_answer',      # Dataset'teki answer key (default: 'interview_answer')
    max_sequence_length=512,            # Max sequence length (default: 512)
    batch_size=8,                       # Batch size for extraction (default: 8)
    return_proba=False,                 # Probability dağılımları isteniyor mu? (default: False)
    use_cache=True,                     # Drive'dan yükleme aktif mi? (default: True)
)
```

### Örnek Kullanım Senaryosu

#### Senaryo 1: İlk Çalıştırma (Test Feature'ları Yok)

```python
# 1. Setup
from src.models.inference import predict_batch_from_dataset
from src.storage.manager import StorageManager

storage = StorageManager(...)
test_ds = storage.load_split('test', task='clarity')

# 2. Model ve tokenizer yükle
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda')

# 3. TF-IDF vectorizer'ı train set'ten fit et
train_ds = storage.load_split('train', task='clarity')
_, _, tfidf_vectorizer = featurize_hf_dataset_in_batches_v2(
    train_ds, tokenizer, model, device,
    tfidf_vectorizer=None  # Fit new TF-IDF
)

# 4. Trained classifier'ı yükle (train+dev üzerinde eğitilmiş)
trained_classifier = ...  # Your trained classifier

# 5. Prediction yap
predictions = predict_batch_from_dataset(
    dataset=test_ds,
    model=model,
    tokenizer=tokenizer,
    classifier=trained_classifier,
    device=device,
    storage_manager=storage,      # ← Checkpoint için gerekli
    model_name='bert',            # ← Checkpoint için gerekli
    task='clarity',               # ← Checkpoint için gerekli
    tfidf_vectorizer=tfidf_vectorizer,
    return_proba=True,
)

# Çıktı:
# "Features not found in Drive, extracting for bert_clarity..."
# [Feature extraction happens...]
# "✓ Saved features to Drive: bert_clarity_test"
# "  Path: features/raw/X_test_bert_clarity.npy"
# "  Next time, features will be loaded from Drive (no re-extraction needed)"
```

#### Senaryo 2: İkinci Çalıştırma (Test Feature'ları Drive'da Var)

```python
# Aynı kod, ama bu sefer:
predictions = predict_batch_from_dataset(
    dataset=test_ds,
    model=model,
    tokenizer=tokenizer,
    classifier=trained_classifier,
    device=device,
    storage_manager=storage,
    model_name='bert',
    task='clarity',
    tfidf_vectorizer=tfidf_vectorizer,
)

# Çıktı:
# "✓ Loaded features from Drive: bert_clarity_test (308 samples)"
# "  Path: features/raw/X_test_bert_clarity.npy"
# [No extraction, directly uses loaded features]
```

### Önemli Notlar

1. **TF-IDF Vectorizer:**
   - Train set'ten fit edilmiş olmalı
   - Test set'te kullanılırken aynı vectorizer kullanılmalı (data leakage önlemek için)
   - Task-dependent ama model-independent (aynı task için tüm modeller aynı TF-IDF'i kullanabilir)

2. **Drive Yapısı:**
   ```
   features/
     raw/
       X_train_{model}_{task}.npy  ← Zaten var (resimde görüyoruz)
       X_dev_{model}_{task}.npy     ← Zaten var (resimde görüyoruz)
       X_test_{model}_{task}.npy     ← Bu fonksiyon oluşturur (yoksa)
   ```

3. **Checkpoint Mekanizması:**
   - `use_cache=True` ve `storage_manager/model_name/task` verilirse:
     - Drive'da test feature'ları varsa: Yükler
     - Drive'da test feature'ları yoksa: Extract eder ve kaydeder
   - `use_cache=False` veya checkpoint bilgileri yoksa:
     - Her seferinde extract eder (Drive'a kaydetmez)

### Workflow Özeti

```
┌─────────────────────────────────────────┐
│ predict_batch_from_dataset() Çağrıldı  │
└──────────────┬──────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Drive'da test       │
    │ feature'ları var mı? │
    └──────┬───────────────┘
           │
    ┌──────┴──────┐
    │             │
   EVET          HAYIR
    │             │
    ▼             ▼
┌─────────┐  ┌──────────────────┐
│ Drive'dan│  │ Feature extraction│
│ yükle    │  │ yap               │
└────┬────┘  └──────┬─────────────┘
     │              │
     │              ▼
     │      ┌──────────────┐
     │      │ Drive'a kaydet│
     │      └──────┬───────┘
     │             │
     └──────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Prediction    │
    │ yap           │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Return        │
    │ predictions   │
    └───────────────┘
```

