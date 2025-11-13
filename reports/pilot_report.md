# CLARITY Pilot Data Report

## 1. Scope
This report captures the exploratory findings and prototype models derived from the two companion notebooks:

- `notebooks/data_analysis.ipynb` – descriptive analysis of the entire QEvasion dataset plus a TF–IDF logistic-regression baseline for the 3-class clarity task.
- `notebooks/advanced_preprocessing.ipynb` – advanced text normalization, structural feature engineering, Hugging Face sentiment augmentation, and a feature-fusion classifier prototype.

All numbers below refer to the raw QEvasion export (`data/raw/QEvasion.csv`, 3,448 rows) unless explicitly noted.

---

## 2. Dataset Characteristics

- **Schema**: 19 raw columns covering interview metadata, question/answer text, GPT-3.5 helper fields, binary prompt flags, and gold labels (`clarity_label`, `evasion_label`).  
- **Time coverage**: 2006–2023 presidential interviews. `interview_year` is used downstream instead of the verbose date string.
- **Label balance**:
  - Clarity (`3,448` rows): `Ambivalent` 2,040 (59.1%), `Clear Reply` 1,052 (30.5%), `Clear Non-Reply` 356 (10.3%).
  - Evasion (top 5 of 9): `Explicit` 1,052, `Dodging` 706, `Implicit` 488, `General` 386, `Deflection` 381.
- **Question structure**: multi-part prompts are rare (86 / 3,448 ≈ 2.5%). The notebook keeps the `has_multiple_questions` boolean for slicing.
- **Length statistics**:
  - Answers: mean 294 words, median 207, 90th percentile 698 → context windows must accommodate very long responses.
  - Questions: shorter (median ≈ 36 words) but often include multiple sentences, lexical overlap with answers averages 0.21, indicating limited verbatim reuse.

---

## 3. Key EDA Findings (`data_analysis.ipynb`)

- **Missingness**: only auxiliary GPT fields show substantial nulls, core question/answer/label columns are complete, so no imputation was required.
- **Distributions**: Clarity and evasion bars (Seaborn) reveal severe class imbalance—`Clear Non-Reply` and long-tail evasions (<150 samples) will require weighted losses or resampling.
- **Answer-length histograms**: the log-scale plot confirms a fat tail beyond 1,500 words, motivating truncation-aware prompting and NE-scrubbing (as described in the main README).
- **Question vs. answer scatter**: mild positive correlation, multi-question prompts (rare blue points) skew toward longer answers but do not form a distinct cluster.
- **Clarity × Evasion heatmap**:
  - `Clear Reply` rows align almost perfectly with the `Explicit` evasion label (a labeling artifact: “Explicit” doubles as the non-evasive clarity case).
  - `Ambivalent` spans `Dodging`, `Implicit`, `General`, `Deflection` in roughly even proportions, supporting the team’s two-stage modeling plan (predict evasion first, then clarity).
- **TF–IDF baseline**:
  - Pipeline: concatenate question+answer → `TfidfVectorizer` (1–2 grams, 25k features, min_df=5) → class-balanced `LogisticRegression`.
  - Split: 80/20 stratified (train 2,758 / test 690).
  - Results (test set): accuracy 0.574, macro-F1 0.53.  
    - `Ambivalent`: F1 0.66 (precision 0.70 / recall 0.62).  
    - `Clear Non-Reply`: F1 0.47 (precision 0.40 / recall 0.56).  
    - `Clear Reply`: F1 0.47 (precision 0.46 / recall 0.48).
  - Takeaway: lexical cues alone capture the dominant `Ambivalent` class but underperform on the minority classes, reinforcing the need for richer features.

---

## 4. Advanced Preprocessing Highlights (`advanced_preprocessing.ipynb`)

### 4.1 Text Normalization
- Lowercases and trims whitespace/noise while **retaining stop words** (per stakeholder request) to preserve pragmatic cues (e.g., hedges, discourse markers).
- Keeps both normalized and original text to support qualitative review.

### 4.2 Structural Features
- Word/character/sentence counts, average sentence length, question-mark flag, hedge frequency (`maybe|perhaps|sort of|kind of|i think`), and question–answer lexical overlap.  
- These features quantify verbosity, evasive hedging, and how directly the answer reuses the question’s vocabulary—signals that correlated with ambiguity in Papantoniou et al. (2024).

### 4.3 Sentiment Augmentation
- Uses the free Hugging Face model `distilbert-base-uncased-finetuned-sst-2-english` to score both questions and answers (positive/negative probabilities).
- Produces eight new numeric features (two sentiments × question/answer) while avoiding GPU reliance by running in CPU batches.
- Serves as a proxy for tone: e.g., negative-question / positive-answer patterns often align with reframing or deflection.

### 4.4 Feature-Fusion Classifier Prototype
- Inputs: TF–IDF (1–2 grams, 30k features) over concatenated normalized text plus the engineered numeric features (scaled via `StandardScaler`).
- Model: class-balanced logistic regression inside a `ColumnTransformer` pipeline.
- Runtime note: full-dataset inference with sentiment scoring is CPU-heavy, for rapid iteration we ran the prototype on a **1,200-sample stratified subset** (train/test = 900/300). This mirrors the notebook’s logic but with manageable compute.
- Results on the 300-sample test split:
  - Accuracy 0.563 (macro-F1 0.52).  
  - `Ambivalent` F1 0.66, `Clear Non-Reply` F1 0.44, `Clear Reply` F1 0.45.
- Interpretation: despite the richer numeric features and sentiment cues, performance remains close to the plain TF–IDF baseline, gains may emerge once the sentiment model is tuned specifically for interview discourse or replaced with the planned evasion logits.
