# SemEval-2024 Task 1: Question-Answer Evasion Detection

A comprehensive machine learning pipeline for detecting question-answer evasion strategies using **25 Context Tree features** extracted from multiple transformer models. This repository implements **four distinct evaluation methodologies** for the SemEval-2024 Task 1 competition.

## 📋 Table of Contents

1. [Overview](#overview)
2. [The 25 Context Tree Features](#the-25-context-tree-features)
3. [Four Evaluation Methodologies](#four-evaluation-methodologies)
4. [Project Structure](#project-structure)
5. [Pipeline Workflow](#pipeline-workflow)
6. [Results and Storage Locations](#results-and-storage-locations)
7. [Core Modules](#core-modules)
8. [Usage](#usage)
9. [Requirements](#requirements)

---

## 📋 Overview

This project implements a comprehensive pipeline for **SemEval-2024 Task 1: Question-Answer Evasion Detection**. The pipeline extracts **25 Context Tree features** from question-answer pairs using six transformer models and performs classification on two main tasks:

### Tasks

1. **Clarity Task** (3 classes):
   - `Clear Reply`: The answer directly addresses the question
   - `Ambivalent`: The answer is ambiguous or partially addresses the question
   - `Clear Non-Reply`: The answer does not address the question

2. **Evasion Task** (9 classes):
   - `Claims ignorance`: "I don't know"
   - `Clarification`: Asking for clarification
   - `Declining to answer`: Explicit refusal
   - `Deflection`: Redirecting to another topic
   - `Dodging`: Avoiding the question
   - `Explicit`: Direct answer
   - `General`: Vague or general response
   - `Implicit`: Indirect answer
   - `Partial/half-answer`: Incomplete answer

### Models

- **BERT** (`bert-base-uncased`)
- **BERT-Political** (`bucketresearch/politicalBiasBERT`)
- **BERT-Ambiguity** (`Slomb/Ambig_Question`)
- **RoBERTa** (`roberta-base`)
- **DeBERTa** (`microsoft/deberta-v3-base`)
- **XLNet** (`xlnet-base-cased`)

### Classifiers

- **LogisticRegression** (with StandardScaler)
- **LinearSVC** (with StandardScaler)
- **RandomForest**
- **MLP** (Multi-Layer Perceptron, with StandardScaler)
- **XGBoost**
- **LightGBM**

---

## 🔬 The 25 Context Tree Features

The pipeline extracts **25 Context Tree features** from each question-answer pair. These features are divided into **7 model-dependent features** (require transformer model) and **18 model-independent features** (text-based only).

### Model-Dependent Features (7 features)

These features require a transformer model to extract attention patterns and tokenization information.

#### 1. `question_model_token_count`
- **Description**: Number of tokens in the question after tokenization by the model
- **Formula**: `count(question_tokens)` where tokens are model-specific (e.g., BERT uses WordPiece)
- **Range**: Integer ≥ 0
- **When it's better**: Higher values indicate longer, more complex questions. Useful for distinguishing between simple yes/no questions and complex multi-part questions.
- **Example**: 
  - Question: "What is your opinion?" → ~5 tokens
  - Question: "Can you explain the detailed process of how this policy was developed and what factors influenced the decision-making?" → ~20 tokens

#### 2. `answer_model_token_count`
- **Description**: Number of tokens in the answer after tokenization
- **Formula**: `count(answer_tokens)`
- **Range**: Integer ≥ 0
- **When it's better**: Longer answers may indicate more detailed responses (Clear Reply) or verbose evasion (General, Dodging). Shorter answers may indicate direct refusals (Declining to answer) or brief clarifications.
- **Example**:
  - Answer: "No comment." → ~2 tokens
  - Answer: "I would need to review the specific details of that policy before providing a comprehensive response." → ~15 tokens

#### 3. `attention_mass_q_to_a_per_qtoken`
- **Description**: Average attention mass from question tokens to answer tokens, normalized by question token count
- **Formula**: 
  ```
  attention_mass_q_to_a_per_qtoken = Σ(attention[q_i, a_j]) / |Q|
  ```
  where `attention[q_i, a_j]` is the attention weight from question token `q_i` to answer token `a_j`, summed over all question-answer token pairs, and `|Q|` is the number of question tokens.
- **Range**: [0, 1] (normalized attention mass)
- **When it's better**: Higher values indicate the model focuses more on the answer when processing the question. This is useful for detecting when answers are relevant to questions (Clear Reply) vs. irrelevant (Clear Non-Reply).
- **Example**: 
  - High value: Question "What is your policy?" and Answer "Our policy is X" → model attends strongly from question to answer
  - Low value: Question "What is your policy?" and Answer "I cannot comment" → model attends weakly

#### 4. `attention_mass_a_to_q_per_atoken`
- **Description**: Average attention mass from answer tokens to question tokens, normalized by answer token count
- **Formula**:
  ```
  attention_mass_a_to_q_per_atoken = Σ(attention[a_i, q_j]) / |A|
  ```
  where `attention[a_i, q_j]` is the attention weight from answer token `a_i` to question token `q_j`, and `|A|` is the number of answer tokens.
- **Range**: [0, 1]
- **When it's better**: Higher values indicate the answer references the question more. Useful for detecting direct answers (Explicit) vs. evasive answers (Dodging, Deflection).
- **Example**:
  - High value: Answer "The policy you asked about is..." → model attends back to question
  - Low value: Answer "I cannot discuss that topic" → model doesn't attend back to question

#### 5. `focus_token_to_answer_strength`
- **Description**: Average maximum attention strength from focus tokens (top-k most central question tokens) to answer tokens
- **Formula**:
  ```
  focus_tokens = top_k(centrality_score, k=min(8, |Q|))
  centrality_score[q_i] = Σ(attention[q_i, :]) + Σ(attention[:, q_i])
  focus_token_to_answer_strength = mean(max(attention[focus_i, a_j] for all a_j))
  ```
  where focus tokens are the top-8 most central question tokens (by incoming + outgoing attention).
- **Range**: [0, 1]
- **When it's better**: Higher values indicate that key question words strongly attend to the answer. Useful for detecting when answers address the core of the question (Clear Reply) vs. peripheral aspects (Partial/half-answer).
- **Example**:
  - High value: Question "What is your policy on X?" (focus: "policy", "X") → Answer "Our policy is..." → strong attention
  - Low value: Question "What is your policy?" → Answer "I need clarification" → weak attention

#### 6. `answer_token_to_focus_strength`
- **Description**: Average maximum attention strength from answer tokens to focus tokens
- **Formula**:
  ```
  answer_token_to_focus_strength = mean(max(attention[a_i, focus_j] for all focus_j))
  ```
- **Range**: [0, 1]
- **When it's better**: Higher values indicate answer tokens attend to key question words. Useful for detecting direct answers (Explicit) vs. evasive answers (Dodging, General).
- **Example**:
  - High value: Answer "The policy is..." attends to question focus "policy"
  - Low value: Answer "I cannot comment" doesn't attend to question focus

#### 7. `focus_token_coverage_ratio`
- **Description**: Fraction of focus tokens that have strong attention (>0.08 threshold) to at least one answer token
- **Formula**:
  ```
  focus_token_coverage_ratio = count(focus_i where max(attention[focus_i, a_j]) > 0.08) / |focus_tokens|
  ```
- **Range**: [0, 1]
- **When it's better**: Higher values indicate more focus tokens are "covered" by the answer. Useful for detecting comprehensive answers (Clear Reply) vs. partial answers (Partial/half-answer, Ambivalent).
- **Example**:
  - High value: All key question words are addressed in the answer
  - Low value: Only some key question words are addressed

---

### Model-Independent Features (18 features)

These features are extracted from text only and are shared across all models.

#### 8. `tfidf_cosine_similarity_q_a`
- **Description**: TF-IDF cosine similarity between question and answer
- **Formula**:
  ```
  tfidf_q = TFIDFVectorizer.transform(question)
  tfidf_a = TFIDFVectorizer.transform(answer)
  similarity = cosine_similarity(tfidf_q, tfidf_a)[0, 0]
  ```
  Uses unigrams and bigrams, English stopwords removed, min_df=2.
- **Range**: [0, 1]
- **When it's better**: Higher values indicate lexical overlap between question and answer. Useful for detecting direct answers (Explicit, Clear Reply) vs. evasive answers with different vocabulary (Dodging, Deflection).
- **Example**:
  - High value: Q: "What is your policy?" A: "Our policy is..." → high lexical overlap
  - Low value: Q: "What is your policy?" A: "I cannot comment on that" → low lexical overlap

#### 9. `content_word_jaccard_q_a`
- **Description**: Jaccard similarity of content words (non-stopwords) between question and answer
- **Formula**:
  ```
  content_words_q = {w for w in question_words if w not in stopwords}
  content_words_a = {w for w in answer_words if w not in stopwords}
  jaccard = |content_words_q ∩ content_words_a| / |content_words_q ∪ content_words_a|
  ```
- **Range**: [0, 1]
- **When it's better**: Higher values indicate shared content vocabulary. Useful for detecting answers that use similar terminology to the question (Clear Reply) vs. answers with different terminology (General, Deflection).
- **Example**:
  - High value: Q: "policy, budget, allocation" A: "policy, budget, allocation" → Jaccard = 1.0
  - Low value: Q: "policy, budget" A: "cannot, comment" → Jaccard = 0.0

#### 10. `question_content_coverage_in_answer`
- **Description**: Fraction of question content words that appear in the answer
- **Formula**:
  ```
  coverage = |content_words_q ∩ content_words_a| / |content_words_q|
  ```
- **Range**: [0, 1]
- **When it's better**: Higher values indicate the answer mentions more question terms. Useful for detecting comprehensive answers (Clear Reply) vs. answers that ignore question terms (Clear Non-Reply, Dodging).
- **Example**:
  - High value: Q: "policy, budget, allocation" → A: "policy, budget, allocation" → coverage = 1.0
  - Low value: Q: "policy, budget" → A: "I cannot comment" → coverage = 0.0

#### 11. `answer_content_word_ratio`
- **Description**: Ratio of content words (non-stopwords) to total words in the answer
- **Formula**:
  ```
  ratio = count(content_words) / count(all_words)
  ```
- **Range**: [0, 1]
- **When it's better**: Higher values indicate more informative, content-rich answers. Useful for detecting substantive answers (Explicit, Clear Reply) vs. formulaic evasions (Declining to answer, Claims ignorance).
- **Example**:
  - High value: "Our policy allocates 50% of the budget to education" → high content word ratio
  - Low value: "I cannot comment on that" → low content word ratio (mostly stopwords)

#### 12. `answer_digit_groups_per_word`
- **Description**: Number of digit groups (consecutive digits) per word in the answer
- **Formula**:
  ```
  digit_groups = count(re.findall(r"\d+", answer))
  words = count(re.findall(r"[A-Za-z']+", answer))
  ratio = digit_groups / max(1, words)
  ```
- **Range**: [0, ∞)
- **When it's better**: Higher values indicate answers with specific numbers/statistics. Useful for detecting detailed answers (Explicit, Clear Reply) vs. vague answers (General, Implicit).
- **Example**:
  - High value: "The budget is $50 million, allocated 30% to education" → 2 digit groups / 10 words = 0.2
  - Low value: "I cannot provide specific numbers" → 0 digit groups

#### 13. `refusal_pattern_match_count`
- **Description**: Count of refusal pattern matches in the answer (case-insensitive regex)
- **Formula**: Count of matches for patterns like:
  - `r"\bI (can't|cannot|won't) (comment|answer|say|discuss)\b"`
  - `r"\bI (don't|do not) (know|have information)\b"`
  - `r"\bno comment\b"`
  - `r"\bI (decline|refuse)\b"`
  - (8 total patterns)
- **Range**: Integer ≥ 0
- **When it's better**: Higher values indicate explicit refusals. Strong indicator for "Declining to answer" and "Claims ignorance" classes.
- **Example**:
  - High value: "I cannot comment on that" → 1 match
  - Low value: "Our policy is..." → 0 matches

#### 14. `clarification_pattern_match_count`
- **Description**: Count of clarification request patterns in the answer
- **Formula**: Count of matches for patterns like:
  - `r"\b(can|could|would) you clarify\b"`
  - `r"\bwhat do you mean\b"`
  - `r"\b(i )?(don'?t|do not) understand\b"`
  - (25+ total patterns)
- **Range**: Integer ≥ 0
- **When it's better**: Higher values indicate clarification requests. Strong indicator for "Clarification" class.
- **Example**:
  - High value: "Can you clarify what you mean?" → 1 match
  - Low value: "Our policy is..." → 0 matches

#### 15. `answer_question_mark_count`
- **Description**: Number of question marks in the answer
- **Formula**: `count("?")` in answer text
- **Range**: Integer ≥ 0
- **When it's better**: Higher values indicate questions in the answer. Useful for detecting clarification requests (Clarification) or rhetorical questions (Dodging, Deflection).
- **Example**:
  - High value: "What do you mean by that?" → 1 question mark
  - Low value: "Our policy is clear." → 0 question marks

#### 16. `answer_word_count`
- **Description**: Total word count in the answer
- **Formula**: `count(re.findall(r"[A-Za-z']+", answer))`
- **Range**: Integer ≥ 0
- **When it's better**: Longer answers may indicate detailed responses (Explicit, Clear Reply) or verbose evasions (General, Dodging). Shorter answers may indicate direct refusals (Declining to answer).
- **Example**:
  - High value: "Our policy allocates 50% of the budget to education, with 30% to healthcare..." → 15 words
  - Low value: "No comment." → 2 words

#### 17. `answer_is_short_question`
- **Description**: Binary feature: 1 if answer has question mark AND ≤10 words, else 0
- **Formula**:
  ```
  answer_is_short_question = 1.0 if (question_mark_count > 0 and word_count <= 10) else 0.0
  ```
- **Range**: {0, 1}
- **When it's better**: Value of 1 indicates short clarification requests. Strong indicator for "Clarification" class.
- **Example**:
  - Value 1: "What do you mean?" → 4 words, 1 question mark
  - Value 0: "Our policy is clear." → no question mark

#### 18. `answer_negation_ratio`
- **Description**: Ratio of negation words to total words in the answer
- **Formula**:
  ```
  negation_words = {"no", "not", "never", "none", "cannot", "don't", "won't", ...}
  ratio = count(negation_words in answer) / count(all_words)
  ```
- **Range**: [0, 1]
- **When it's better**: Higher values indicate negative statements. Useful for detecting refusals (Declining to answer, Claims ignorance) vs. affirmative answers (Explicit).
- **Example**:
  - High value: "I cannot comment on that" → 1 negation / 4 words = 0.25
  - Low value: "Our policy is clear" → 0 negations

#### 19. `answer_hedge_ratio`
- **Description**: Ratio of hedge words (uncertainty markers) to total words
- **Formula**:
  ```
  hedge_words = {"maybe", "perhaps", "probably", "seems", "appears", "roughly", "about", ...}
  ratio = count(hedge_words in answer) / count(all_words)
  ```
- **Range**: [0, 1]
- **When it's better**: Higher values indicate uncertain or vague language. Useful for detecting ambiguous answers (Ambivalent, Implicit, General) vs. definitive answers (Explicit, Clear Reply).
- **Example**:
  - High value: "It seems like the policy might be around 50%" → 2 hedges / 8 words = 0.25
  - Low value: "The policy is 50%" → 0 hedges

#### 20. `question_sentiment_polarity`
- **Description**: Sentiment polarity of the question (positive - negative)
- **Formula**:
  ```
  sentiment_scores = sentiment_pipeline(question)  # Returns {positive: 0.7, negative: 0.2, neutral: 0.1}
  polarity = positive_score - negative_score
  ```
  Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` model.
- **Range**: [-1, 1]
- **When it's better**: Positive values indicate positive sentiment questions, negative values indicate negative sentiment. Useful for detecting how question tone affects answer style (e.g., negative questions may elicit defensive answers).
- **Example**:
  - Positive: "What are the benefits of this policy?" → polarity ≈ 0.5
  - Negative: "Why is this policy so problematic?" → polarity ≈ -0.5

#### 21. `answer_sentiment_polarity`
- **Description**: Sentiment polarity of the answer (positive - negative)
- **Formula**: Same as question sentiment polarity, applied to answer text
- **Range**: [-1, 1]
- **When it's better**: Positive values indicate positive sentiment answers. Useful for detecting defensive or evasive answers (negative polarity) vs. cooperative answers (positive polarity).
- **Example**:
  - Positive: "Our policy is beneficial and well-received" → polarity ≈ 0.6
  - Negative: "I cannot comment on that problematic issue" → polarity ≈ -0.3

#### 22. `answer_char_per_sentence`
- **Description**: Average characters per sentence in the answer
- **Formula**:
  ```
  sentences = re.split(r"(?<=[.!?])\s+", answer)
  char_per_sentence = total_characters / max(1, len(sentences))
  ```
- **Range**: [0, ∞)
- **When it's better**: Higher values indicate longer, more complex sentences. Useful for detecting detailed explanations (Explicit, Clear Reply) vs. brief responses (Declining to answer, Claims ignorance).
- **Example**:
  - High value: "Our policy allocates 50% of the budget to education, with 30% to healthcare, and the remaining 20% to infrastructure." → 1 sentence, 120 chars → 120 chars/sentence
  - Low value: "No comment." → 1 sentence, 10 chars → 10 chars/sentence

#### 23. `inaudible`
- **Description**: Binary metadata feature indicating if the question/answer was inaudible
- **Formula**: Directly from dataset metadata (boolean)
- **Range**: {0, 1}
- **When it's better**: Value of 1 indicates audio quality issues. May correlate with "Clarification" requests or incomplete answers.
- **Example**: From dataset metadata

#### 24. `multiple_questions`
- **Description**: Binary metadata feature indicating if the question contains multiple sub-questions
- **Formula**: Directly from dataset metadata (boolean)
- **Range**: {0, 1}
- **When it's better**: Value of 1 indicates complex questions. May correlate with "Partial/half-answer" (only some sub-questions answered) or "Clarification" requests.
- **Example**: From dataset metadata

#### 25. `affirmative_questions`
- **Description**: Binary metadata feature indicating if the question is affirmative (yes/no style)
- **Formula**: Directly from dataset metadata (boolean)
- **Range**: {0, 1}
- **When it's better**: Value of 1 indicates yes/no questions. May correlate with direct answers (Explicit) vs. evasive answers (Dodging, General).
- **Example**: From dataset metadata

---

## 🎯 Three Final Evaluation Methodologies

This repository implements **three distinct final evaluation methodologies** on the **test set** (held-out, never used for training or development). Each methodology uses different feature selection strategies and ensemble approaches.

**Experimental Setup**: 
- **6 Transformer Models**: BERT, BERT-Political, BERT-Ambiguity, RoBERTa, DeBERTa, XLNet
- **6 Classifiers**: LogisticRegression, LinearSVC, RandomForest, MLP, XGBoost, LightGBM
- **2 Tasks**: Clarity (3 classes), Evasion (9 classes)

---

### Methodology 1: Individual Model Baseline Evaluation

**Notebook**: `05_final_evaluation.py`  
**Description**: Baseline evaluation where each of the 6 models is evaluated separately with each of the 6 classifiers, using all 25 features per model. No feature selection or ensemble.

**Experimental Design**: 
- **6 models** × **6 classifiers** × **2 tasks** = **72 model-classifier-task combinations**
- Each model uses its own 25 features (7 model-dependent + 18 model-independent)

**Process**:
1. Extract 25 features for each model on test set
2. Train on Train+Dev combined data (final training)
3. Evaluate each model×classifier combination on test set
4. Generate results tables and visualizations

**Feature Selection**: None (uses all 25 features per model)

**Ensemble Strategy**: None (individual model results only)

**Evaluation Set**: Test set (308 samples for Clarity, 275 samples for Evasion)

**Results Location** (Google Drive):
- **Predictions**: `results/FinalResultsType1/predictions/pred_test_{model}_{classifier}_{task}.npy`
- **Probabilities**: `results/FinalResultsType1/probabilities/probs_test_{model}_{classifier}_{task}.npy`
- **Plots**: `results/FinalResultsType1/plots/{model}_{task}_{classifier}/`
- **Tables**: `results/FinalResultsType1/tables/`
- **Metadata**: `results/FinalResultsType1Results/FINAL_TEST_{model}_{task}.json`

**Total Results**: **72 model-classifier-task combinations**

---

### Methodology 2: Ablation Study with Classifier-Specific Feature Selection and Weighted Ensemble

**Notebook**: `03_5_ablation_study.py`  
**Description**: Comprehensive ablation study with classifier-specific feature selection. Each classifier gets 40 features selected via greedy forward selection from 60 early fusion features. This methodology produces **three distinct result types**: (1) individual classifier hard label predictions, (2) same hard labels as separate evaluation, and (3) weighted average ensemble from probabilities.

**Prerequisite Step (Development Evaluation)**: 
- **Notebook**: `03_train_evaluate.ipynb`
- **Purpose**: Development set evaluation (not final evaluation) used for model selection and to guide feature selection
- **Process**: Train on train set, evaluate on dev set (6 models × 6 classifiers × 2 tasks)
- **Note**: This is **not a separate methodology** but rather a development step that precedes Methodology 2

**Experimental Design**:
- **60 Early Fusion Features**: 18 model-independent + 42 model-dependent (6 models × 7 features)
- **40 Features per Classifier**: Selected via greedy forward selection (global top 20 + classifier-specific greedy 20)
- **6 classifiers** × **2 tasks** = **12 classifier-task combinations** for individual results
- **2 tasks** = **2 ensemble results** (one per task)

**Process**:

#### Step 1: Development Set Evaluation (Prerequisite)
- Extract 25 features for each model on train and dev sets
- Train on train set, evaluate on dev set
- Compare classifier performance to guide feature selection
- **Output**: `predictions/pred_dev_{model}_{classifier}_{task}.npy`

#### Step 2: Single-Feature Ablation
- Evaluate each of the 25 features individually across all 36 model×classifier combinations
- Compute statistics: min, median, mean, std, max F1
- Calculate weighted score: `0.5 * mean_f1 + 0.3 * best_f1 + 0.2 * (1 - normalized_std)`
- Rank features by weighted score

#### Step 3: Early Fusion Feature Creation (60 features)
- **18 model-independent features** (shared across all models)
- **42 model-dependent features** (6 models × 7 features each)
- Total: **60 features** via concatenation

#### Step 4: Classifier-Specific Feature Selection
- **Global Top 20**: Selected from weighted score ranking (across all models)
- **Classifier-Specific Greedy 20**: Greedy forward selection for each classifier (starts with global top 20, adds up to 20 more)
- **Final Feature Set**: 40 features per classifier (global 20 + classifier-specific greedy 20)

#### Step 5: Training and Evaluation on Test Set
- Train on Train+Dev combined data
- Evaluate on **test set** (final evaluation)
- Each classifier uses its own 40 selected features

#### Step 6: Three Result Types

**Result Type 1: Individual Classifier Hard Label Predictions**
- Each classifier produces hard label predictions using its own 40 selected features
- **6 classifiers** × **2 tasks** = **12 classifier-task combinations**
- **Output**: `predictions/{classifier}_{task}_predictions.npy`

**Result Type 2: Same Hard Labels (Separate Evaluation)**
- Same 12 hard label predictions from Result Type 1, but evaluated separately as individual classifier results
- Used for comparing individual classifier performance
- **Output**: Same files as Result Type 1, but with separate metrics

**Result Type 3: Weighted Average Ensemble from Probabilities**
- Collect probabilities from all 6 classifiers
- Weight each classifier by its Macro F1 score on test set: `weight_i = MacroF1_i / Σ(MacroF1_j)`
- Compute weighted average: `ensemble_proba = Σ(weight_i * proba_i)`
- Generate hard labels from weighted average probabilities: `argmax(ensemble_proba)`
- **2 tasks** = **2 ensemble results** (one per task)
- **Output**: 
  - `predictions/ensemble_hard_labels_from_weighted_proba_{task}.npy`
  - `probabilities/ensemble_weighted_average_probabilities_{task}.npy`

**Feature Selection**: 
- **40 features per classifier** (selected via greedy forward selection)
- Global top 20 + classifier-specific greedy 20

**Ensemble Strategy**: 
- **Result Type 1 & 2**: Individual classifier hard label predictions (no ensemble)
- **Result Type 3**: Weighted average ensemble from probabilities (weights = Macro F1 scores)

**Evaluation Set**: Test set (final evaluation)

**Results Location** (Google Drive):
- **Ablation Results**: `results/FinalResultsType2/ablation/`
  - `single_feature_{task}.csv` (raw ablation results)
  - `feature_ranking_{task}.csv` (feature rankings with statistics)
  - `selected_features_all.json` (classifier-specific feature selections)
  - `greedy_trajectory_{model}_{task}_{classifier}.csv` (greedy selection trajectories)
- **Result Type 1 & 2 (Individual Classifier Hard Labels)**: `results/FinalResultsType2/classifier_specific/`
  - **Hard Labels**: `predictions/{classifier}_{task}_predictions.npy`
  - **Probabilities**: `probabilities/{classifier}_{task}_probabilities.npy`
  - **Metrics**: `metrics/ensemble_evaluation_metrics_{task}.json`
- **Result Type 3 (Weighted Average Ensemble)**: `results/FinalResultsType2/classifier_specific/`
  - **Hard Labels from Weighted Proba**: `predictions/ensemble_hard_labels_from_weighted_proba_{task}.npy`
  - **Weighted Average Probabilities**: `probabilities/ensemble_weighted_average_probabilities_{task}.npy`
  - **Ensemble Weights**: `metrics/ensemble_classifier_weights_{task}.json`

**Total Results**:
- **Result Type 1 & 2**: **12 classifier-task combinations** (6 classifiers × 2 tasks)
- **Result Type 3**: **2 ensemble results** (one per task)
- **Total**: **14 distinct result sets** (12 individual + 2 ensemble)

**Key Innovation**: This methodology uses **classifier-specific feature selection**, meaning each classifier gets features optimized for its own learning algorithm (e.g., RandomForest may prefer different features than LogisticRegression). The three result types allow comprehensive evaluation: individual classifier performance (Type 1 & 2) and ensemble performance (Type 3).

---

### Methodology 3: Early Fusion Baseline Evaluation

**Notebook**: `0_4_early_fusion.py`  
**Description**: Early fusion of all 60 features (18 model-independent + 42 model-dependent from 6 models) evaluated with all 6 classifiers. No feature selection - uses all 60 features. Baseline for comparison with Methodology 2's feature selection approach.

**Process**:

#### Step 1: Early Fusion Feature Creation
- **18 model-independent features** (extracted once, shared)
- **42 model-dependent features** (6 models × 7 features each)
- Concatenation: `[18 model-independent | 42 model-dependent]` = **60 features**

#### Step 2: Training and Evaluation
- Train on Train+Dev combined data
- Evaluate on **test set** (final evaluation)
- All classifiers use the same 60 features (no feature selection)

**Feature Selection**: None (uses all 60 features)

**Ensemble Strategy**: None (individual classifier results only)

**Evaluation Set**: Test set (final evaluation)

**Results Location** (Google Drive):
- **Test Features**: `results/FinalResultsType3/test/X_test_60feat_{task}.npy`
- **Predictions**: `results/FinalResultsType3/predictions/pred_test_{classifier}_{task}.npy`
- **Probabilities**: `results/FinalResultsType3/predictions/proba_test_{classifier}_{task}.npy`
- **Tables**: `results/FinalResultsType3/tables/`
  - `summary_{task}.csv` (per-task summary)
  - `summary_all_tasks_pivot.csv` (classifier × task pivot table)
  - `final_summary_classifier_wise.csv` (includes hierarchical evaluations)
  - `final_summary_detailed.csv` (all metrics)
- **Results**: `results/FinalResultsType3/final_results_type3.json`

**Experimental Design**: 
- **60 Early Fusion Features**: 18 model-independent + 42 model-dependent (6 models × 7 features)
- **6 classifiers** × **2 tasks** = **12 classifier-task combinations**
- All classifiers use the same 60 features (no feature selection)

**Total Results**: **12 classifier-task combinations** (6 classifiers × 2 tasks)

**Additional Evaluations**:
- **Hierarchical Evasion → Clarity**: Maps evasion predictions to clarity labels
- **Annotator-Based Clarity**: Evaluates against annotator1/2/3 clarity labels (mapped from evasion)

**Key Innovation**: This methodology uses **early fusion** (feature-level concatenation) to combine information from all 6 models into a single 60-dimensional feature vector, allowing classifiers to learn cross-model patterns. Serves as baseline for comparison with Methodology 2's classifier-specific feature selection (40 features per classifier).

---

## 📊 Methodology Comparison

| Methodology | Features | Feature Selection | Ensemble | Evaluation Set | Total Results |
|------------|----------|------------------|----------|----------------|---------------|
| **Method 1: Individual Model Baseline** | 25 per model | None | None | Test | 72 combinations (6 models × 6 classifiers × 2 tasks) |
| **Method 2: Ablation Study** | 40 per classifier | Greedy (global 20 + classifier-specific 20) | Weighted average (Result Type 3) | Test | 14 result sets (12 individual + 2 ensemble) |
| **Method 3: Early Fusion Baseline** | 60 (early fusion) | None | None | Test | 12 combinations (6 classifiers × 2 tasks) |

**Note**: Development set evaluation (`03_train_evaluate.ipynb`) is a prerequisite step for Methodology 2, not a separate methodology. It evaluates 6 models × 6 classifiers × 2 tasks on the development set for model selection and feature selection guidance.

---

## 🏗️ Project Structure

```
semeval-context-tree-modular/
├── notebooks/              # Main pipeline notebooks
│   ├── 00_setup.ipynb                    # Repository setup and Drive mount
│   ├── 01_data_split.ipynb               # Dataset splitting (Train/Dev/Test)
│   ├── 02_feature_extraction_separate.ipynb  # Feature extraction per model
│   ├── 03_train_evaluate.ipynb            # Prerequisite: Dev set evaluation (for Methodology 2)
│   ├── 03_5_ablation_study.py             # Methodology 2: Ablation study with classifier-specific selection
│   ├── 0_4_early_fusion.py               # Methodology 3: Early fusion baseline
│   └── 05_final_evaluation.py             # Methodology 1: Individual model baseline
├── src/                    # Python source code
│   ├── data/               # Dataset loading and splitting
│   ├── features/           # Feature extraction and fusion
│   │   ├── extraction.py  # 25 feature extraction functions
│   │   ├── fusion.py       # Early fusion functions
│   │   └── utils.py        # Feature calculation utilities
│   ├── models/             # Model training and classifiers
│   │   ├── classifiers.py # Classifier definitions
│   │   ├── trainer.py     # Training functions
│   │   └── hierarchical.py # Evasion → Clarity mapping
│   ├── evaluation/         # Metrics and visualization
│   │   ├── metrics.py     # Metric computation
│   │   ├── tables.py      # Results table generation
│   │   └── visualizer.py  # Plotting functions
│   └── storage/            # Data storage management
│       └── manager.py     # StorageManager class
├── scripts/                # Utility scripts
└── metadata/               # Feature and result metadata
```

---

## 🔄 Pipeline Workflow

### 1. Setup (`00_setup.ipynb`)
- Clone repository from GitHub
- Mount Google Drive for persistent storage
- Initialize StorageManager

### 2. Data Splitting (`01_data_split.ipynb`)
- Load QEvasion dataset from HuggingFace (`ailsntua/QEvasion`)
- **Clarity Task**: Use all samples (no filtering)
- **Evasion Task**: Apply majority voting (keep samples with ≥2/3 annotator agreement)
- Split HuggingFace train into Train (80%) and Dev (20%)
- Keep HuggingFace test untouched (only used in final evaluation)

**Outputs**:
- `splits/dataset_splits_{task}.pkl` (task-specific splits)

### 3. Feature Extraction (`02_feature_extraction_separate.ipynb`)
Extract 25 Context Tree features for each transformer model separately.

**Outputs**:
- `features/raw/X_{split}_{model}_{task}.npy` (feature matrices)
- `metadata/features_{split}_{model}_{task}.json` (feature metadata)

### 4. Development Set Evaluation (`03_train_evaluate.ipynb`) - Prerequisite for Methodology 2
Train and evaluate on development set (for model selection and feature selection guidance). This is **not a final evaluation** but a prerequisite step.

**Outputs**:
- Predictions: `predictions/pred_dev_{model}_{classifier}_{task}.npy`
- Probabilities: `features/probabilities/probs_dev_{model}_{classifier}_{task}.npy`
- Results: `results/{model}_{task}_separate.json`

### 5. Ablation Study (`03_5_ablation_study.py`) - Methodology 2
Comprehensive feature ablation and classifier-specific feature selection with three result types.

**Outputs**:
- Ablation results: `results/FinalResultsType2/ablation/`
- Individual classifier results (Type 1 & 2): `results/FinalResultsType2/classifier_specific/predictions/`
- Weighted ensemble results (Type 3): `results/FinalResultsType2/classifier_specific/predictions/ensemble_*.npy`

### 6. Early Fusion (`0_4_early_fusion.py`) - Methodology 3
Early fusion of 60 features evaluated with all classifiers.

**Outputs**:
- Fused features: `results/FinalResultsType3/test/X_test_60feat_{task}.npy`
- Predictions: `results/FinalResultsType3/predictions/`
- Tables: `results/FinalResultsType3/tables/`

### 7. Final Evaluation (`05_final_evaluation.py`) - Methodology 1
Evaluate all model×classifier combinations on test set.

**Outputs**:
- Test predictions: `results/FinalResultsType1/predictions/`
- Test probabilities: `results/FinalResultsType1/probabilities/`
- Final tables: `results/FinalResultsType1/tables/`

---

## 📁 Results and Storage Locations

**IMPORTANT: All experimental results, logs, predictions, checkpoints, models, features, plots, and analysis outputs are available in the following Google Drive folder:**

**Google Drive Results Folder**: https://drive.google.com/drive/folders/1P2ugCvV6LStQX5FZ_gQBKVN07L4_lizb?usp=sharing

This folder contains:
- **checkpoints**: Model checkpoints and saved model states
- **features**: Extracted feature matrices and feature metadata
- **models**: Trained model files
- **paper**: Paper-related documents and drafts
- **plots**: Visualization plots and figures
- **predictions**: Prediction outputs for all methodologies
- **results**: Comprehensive results tables, metrics, and evaluation outputs
- **splits**: Dataset splits (train/dev/test) for both Clarity and Evasion tasks

All results are also stored in **Google Drive** at `/content/drive/MyDrive/semeval_data/` (when running in Colab).

### Methodology 1 (Basic Evaluation)
**Drive Path**: `results/FinalResultsType1/`
- Predictions: `predictions/pred_test_{model}_{classifier}_{task}.npy`
- Probabilities: `probabilities/probs_test_{model}_{classifier}_{task}.npy`
- Plots: `plots/{model}_{task}_{classifier}/`
- Tables: `tables/`
- Metadata: `results/FinalResultsType1Results/FINAL_TEST_{model}_{task}.json`

### Development Set Evaluation (Prerequisite for Methodology 2)
**Drive Path**: `results/` and `predictions/`
- Predictions: `predictions/pred_dev_{model}_{classifier}_{task}.npy`
- Probabilities: `features/probabilities/probs_dev_{model}_{classifier}_{task}.npy`
- Results: `results/{model}_{task}_separate.json`

### Methodology 2 (Ablation Study with Classifier-Specific Selection)
**Drive Path**: `results/FinalResultsType2/`
- **Ablation**: `ablation/`
  - `single_feature_{task}.csv`
  - `feature_ranking_{task}.csv`
  - `selected_features_all.json`
  - `greedy_trajectory_{model}_{task}_{classifier}.csv`
- **Classifier-Specific**: `classifier_specific/`
  - Hard labels: `predictions/{classifier}_{task}_predictions.npy`
  - Probabilities: `probabilities/{classifier}_{task}_probabilities.npy`
  - Metrics: `metrics/ensemble_evaluation_metrics_{task}.json`
- **Ensemble**: `classifier_specific/`
  - Hard labels: `predictions/ensemble_hard_labels_from_weighted_proba_{task}.npy`
  - Probabilities: `probabilities/ensemble_weighted_average_probabilities_{task}.npy`
  - Weights: `metrics/ensemble_classifier_weights_{task}.json`

### Methodology 3 (Early Fusion Baseline)
**Drive Path**: `results/FinalResultsType3/`
- Test features: `test/X_test_60feat_{task}.npy`
- Predictions: `predictions/pred_test_{classifier}_{task}.npy`
- Probabilities: `predictions/proba_test_{classifier}_{task}.npy`
- Tables: `tables/`
  - `summary_{task}.csv`
  - `summary_all_tasks_pivot.csv`
  - `final_summary_classifier_wise.csv`
  - `final_summary_detailed.csv`
- Results: `final_results_type3.json`

---

## 🧩 Core Modules

### `src/storage/manager.py` - StorageManager
Manages data storage: GitHub for metadata, Google Drive for large files.

**Key Methods**:
- `save_features()` / `load_features()`: Feature matrices
- `save_splits()` / `load_split()`: Dataset splits (task-specific)
- `save_predictions()` / `load_predictions()`: Hard label predictions
- `save_probabilities()` / `load_probabilities()`: Probability distributions
- `save_fused_features()` / `load_fused_features()`: Early fusion features

### `src/features/extraction.py` - Feature Extraction
Extracts 25 Context Tree features from question-answer pairs.

**Key Functions**:
- `extract_batch_features_v2()`: Extract all 25 features from a batch
- `featurize_hf_dataset_in_batches_v2()`: Process entire dataset in batches
- `featurize_model_independent_features()`: Extract 18 model-independent features
- `featurize_model_dependent_features()`: Extract 7 model-dependent features
- `get_feature_names()`: Returns list of 25 feature names
- `get_model_independent_feature_names()`: Returns 18 model-independent feature names
- `get_model_dependent_feature_names()`: Returns 7 model-dependent feature names

### `src/models/classifiers.py` - Classifiers
Defines classifier instances with proper preprocessing.

**Classifiers**: LogisticRegression, LinearSVC, RandomForest, MLP, XGBoost, LightGBM

**Note**: Uses LabelEncoder to convert string labels to numeric (required for MLP, XGBoost, LightGBM)

### `src/models/hierarchical.py` - Hierarchical Approach
Maps evasion predictions to clarity labels using hierarchical mapping.

**Mapping**:
- **Non-Reply** → "Clear Non-Reply": Claims ignorance, Clarification, Declining to answer
- **Reply** → "Clear Reply": Explicit
- **Others** → "Ambivalent": Implicit, Dodging, General, Deflection, Partial/half-answer

### `src/features/fusion.py` - Feature Fusion
Implements early fusion by concatenating features from multiple models.

**Function**: `fuse_attention_features()` - Horizontal concatenation

---

## 📝 Usage

1. **Setup**: Run `00_setup.ipynb` to clone repository and mount Google Drive
2. **Data Splitting**: Run `01_data_split.ipynb` to create train/dev/test splits
3. **Feature Extraction**: Run `02_feature_extraction_separate.ipynb` to extract 25 features for each model
4. **Development Evaluation**: Run `03_train_evaluate.ipynb` (prerequisite for Methodology 2) for dev set results
5. **Ablation Study**: Run `03_5_ablation_study.py` (Methodology 2) for classifier-specific feature selection with three result types
6. **Early Fusion**: Run `0_4_early_fusion.py` (Methodology 3) for 60-feature early fusion baseline
7. **Final Evaluation**: Run `05_final_evaluation.py` (Methodology 1) for individual model baseline evaluation

**⚠️ CRITICAL**: Test set is **ONLY** accessed in final evaluation notebooks (Methodologies 1, 2, and 3). Do not access test set in development notebooks.

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- Transformers (HuggingFace)
- scikit-learn
- pandas, numpy
- XGBoost, LightGBM
- Google Colab (for Drive integration) or local setup with Google Drive API

## 🔑 Key Features

1. **Task-Specific Splits**: Clarity and Evasion have different splits (Evasion is filtered by majority voting)
2. **Test Set Isolation**: Test set is only accessed in final evaluation (prevents data leakage)
3. **TF-IDF Per Task**: Each task gets its own TF-IDF vectorizer (fitted on train, applied to dev/test)
4. **Label Encoding**: String labels converted to numeric (required for some classifiers)
5. **Storage Management**: Large files in Google Drive, metadata in GitHub
6. **Reproducibility**: All random seeds fixed to 42
   - Python random, NumPy, PyTorch, and HuggingFace Transformers seeds set to 42
   - PyTorch deterministic mode enabled for full reproducibility
   - Classifiers use `random_state=42`
   - Data splits use `seed=42`
   - Each notebook has a reproducibility setup cell that must be run first

---

## 📄 License

See LICENSE file for details.

---

## 🙏 Acknowledgments

- SemEval-2024 Task 1 organizers
- QEvasion dataset creators
- HuggingFace for transformer models

---

**Project**: SemEval-2024 Task 1 - Question-Answer Evasion Detection  
**Repository**: https://github.com/EonTechie/semeval-context-tree-modular
