"""
Main feature extraction functions
"""
import torch
import numpy as np
import re
from tqdm import tqdm
from typing import List, Tuple, Optional, Any, Dict
from .utils import (
    split_segments_safe,
    compute_tfidf_cosine_similarity_per_pair,
    compute_content_word_set_metrics,
    content_word_ratio,
    digit_group_count,
    extract_pattern_based_features_case_insensitive,
    extract_answer_lexicon_ratios
)


def get_feature_names() -> List[str]:
    """
    Get the list of 25 Context Tree feature names.
    
    This function returns the feature names in the correct order, matching
    the order of features returned by extract_batch_features_v2.
    
    Returns:
        List of 25 feature names (same for all models)
    """
    return [
        # Model-dependent features (7)
        "question_model_token_count",
        "answer_model_token_count",
        "attention_mass_q_to_a_per_qtoken",
        "attention_mass_a_to_q_per_atoken",
        "focus_token_to_answer_strength",
        "answer_token_to_focus_strength",
        "focus_token_coverage_ratio",
        # Model-independent features (18)
        "tfidf_cosine_similarity_q_a",
        "content_word_jaccard_q_a",
        "question_content_coverage_in_answer",
        "answer_content_word_ratio",
        "answer_digit_groups_per_word",
        "refusal_pattern_match_count",
        "clarification_pattern_match_count",
        "answer_question_mark_count",
        "answer_word_count",
        "answer_is_short_question",
        "answer_negation_ratio",
        "answer_hedge_ratio",
        # New sentiment features (2)
        "question_sentiment_polarity",
        "answer_sentiment_polarity",
        # New structural features (1)
        "answer_char_per_sentence",
        # New metadata features (3)
        "inaudible",
        "multiple_questions",
        "affirmative_questions",
    ]


def get_model_independent_feature_names() -> List[str]:
    """
    Get the list of 18 model-independent feature names.
    
    These features are extracted once and reused for all models.
    
    Returns:
        List of 18 model-independent feature names
    """
    return [
        # TF-IDF (1)
        "tfidf_cosine_similarity_q_a",
        # Content word (3)
        "content_word_jaccard_q_a",
        "question_content_coverage_in_answer",
        "answer_content_word_ratio",
        # Pattern (5)
        "refusal_pattern_match_count",
        "clarification_pattern_match_count",
        "answer_question_mark_count",
        "answer_word_count",
        "answer_is_short_question",
        # Lexicon (2)
        "answer_negation_ratio",
        "answer_hedge_ratio",
        # Digit (1)
        "answer_digit_groups_per_word",
        # Sentiment (2) - NEW
        "question_sentiment_polarity",
        "answer_sentiment_polarity",
        # Structural (1) - NEW
        "answer_char_per_sentence",
        # Metadata (3) - NEW
        "inaudible",
        "multiple_questions",
        "affirmative_questions",
    ]


def get_model_dependent_feature_names() -> List[str]:
    """
    Get the list of 7 model-dependent feature names.
    
    These features are extracted separately for each model.
    
    Returns:
        List of 7 model-dependent feature names
    """
    return [
        "question_model_token_count",
        "answer_model_token_count",
        "attention_mass_q_to_a_per_qtoken",
        "attention_mass_a_to_q_per_atoken",
        "focus_token_to_answer_strength",
        "answer_token_to_focus_strength",
        "focus_token_coverage_ratio",
    ]


def extract_batch_features_v2(
    question_text_list: List[str],
    answer_text_list: List[str],
    tokenizer,
    model,
    device,
    max_sequence_length: int = 512,
    tfidf_vectorizer=None,
    debug_segment_split: bool = False,
    max_debug_print: int = 2,
    sentiment_pipeline=None,
    metadata_dict: Optional[Dict[str, List]] = None,
) -> Tuple[np.ndarray, List[str], Any]:
    """
    Extract 25 Context Tree features from a batch of (question, answer) pairs
    
    Args:
        question_text_list: List of question texts
        answer_text_list: List of answer texts
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model (must support output_attentions=True)
        device: torch device
        max_sequence_length: Max sequence length for tokenization
        tfidf_vectorizer: Pre-fitted TF-IDF vectorizer (or None to fit new)
        debug_segment_split: Enable debug output for segment splitting
        max_debug_print: Max number of debug prints
        sentiment_pipeline: Optional HuggingFace sentiment pipeline (for sentiment features)
        metadata_dict: Optional dict with metadata columns {'inaudible': [...], 'multiple_questions': [...], 'affirmative_questions': [...]}
    
    Returns:
        feature_matrix: (N, 25) numpy array
        feature_names: List of 25 feature names
        tfidf_vectorizer: Fitted TF-IDF vectorizer (for reuse)
    """
    # Tokenize
    model_inputs = tokenizer(
        question_text_list,
        answer_text_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_sequence_length,
    )
    
    # Move to device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    # Forward pass (with attentions)
    with torch.no_grad():
        model_outputs = model(**model_inputs, output_attentions=True)
    
    # Last-layer attention averaged over heads: (B, Heads, T, T) -> (B, T, T)
    last_layer_attention = model_outputs.attentions[-1].mean(dim=1)
    
    # Split question/answer segments
    question_position_mask, answer_position_mask = split_segments_safe(
        model_inputs,
        tokenizer,
        debug=debug_segment_split,
        max_print=max_debug_print,
    )
    
    batch_size, sequence_length, _ = last_layer_attention.shape
    
    # Compute TF-IDF cosine similarity
    tfidf_cosine_sim, tfidf_vectorizer = compute_tfidf_cosine_similarity_per_pair(
        question_text_list,
        answer_text_list,
        fitted_vectorizer=tfidf_vectorizer,
    )
    
    feature_rows = []
    
    for example_index in range(batch_size):
        # Token positions
        question_token_positions = question_position_mask[example_index].nonzero(as_tuple=True)[0]
        answer_token_positions = answer_position_mask[example_index].nonzero(as_tuple=True)[0]
        
        # Model token counts
        question_model_token_count = int(len(question_token_positions))
        answer_model_token_count = int(len(answer_token_positions))
        
        # Defaults for attention features
        attention_mass_q_to_a_per_qtoken = 0.0
        attention_mass_a_to_q_per_atoken = 0.0
        focus_token_to_answer_strength = 0.0
        answer_token_to_focus_strength = 0.0
        focus_token_coverage_ratio = 0.0
        
        if question_model_token_count > 0 and answer_model_token_count > 0:
            # Attention submatrices
            q_to_a_attention = last_layer_attention[example_index][question_token_positions][:, answer_token_positions]
            a_to_q_attention = last_layer_attention[example_index][answer_token_positions][:, question_token_positions]
            
            # Normalized attention mass
            attention_mass_q_to_a_per_qtoken = float(q_to_a_attention.sum().item() / max(1, question_model_token_count))
            attention_mass_a_to_q_per_atoken = float(a_to_q_attention.sum().item() / max(1, answer_model_token_count))
            
            # Focus token selection (top-k by centrality)
            question_outgoing_sum = last_layer_attention[example_index][question_token_positions].sum(dim=1)
            question_incoming_sum = last_layer_attention[example_index][:, question_token_positions].sum(dim=0)
            question_token_centrality_score = question_outgoing_sum + question_incoming_sum
            
            max_focus_token_count = min(8, question_model_token_count)
            top_focus_indices_within_question = torch.topk(
                question_token_centrality_score,
                k=max_focus_token_count
            ).indices
            focus_token_positions = question_token_positions[top_focus_indices_within_question]
            
            # Focus -> answer attention
            focus_to_answer_attention = last_layer_attention[example_index][focus_token_positions][:, answer_token_positions]
            answer_to_focus_attention = last_layer_attention[example_index][answer_token_positions][:, focus_token_positions]
            
            # Focus features
            focus_token_to_answer_strength = float(focus_to_answer_attention.max(dim=1).values.mean().item())
            answer_token_to_focus_strength = float(answer_to_focus_attention.max(dim=1).values.mean().item())
            
            # Focus coverage
            attention_threshold = 0.08
            focus_token_coverage_ratio = float(
                (focus_to_answer_attention.max(dim=1).values > attention_threshold).float().mean().item()
            )
        
        # Content word metrics
        content_jaccard_q_a, question_content_coverage_in_answer = compute_content_word_set_metrics(
            question_text_list[example_index],
            answer_text_list[example_index],
        )
        
        # Answer content word ratio
        answer_content_word_ratio = float(content_word_ratio(answer_text_list[example_index]))
        
        # Digit groups per word
        raw_digit_group_count = digit_group_count(answer_text_list[example_index])
        _, _, _, answer_word_count = extract_pattern_based_features_case_insensitive(answer_text_list[example_index])
        answer_digit_groups_per_word = float(raw_digit_group_count / max(1, answer_word_count))
        
        # Pattern-based features
        refusal_pattern_match_count, clarification_pattern_match_count, answer_question_mark_count, answer_word_count = (
            extract_pattern_based_features_case_insensitive(answer_text_list[example_index])
        )
        
        # Short question style
        answer_is_short_question = float((answer_question_mark_count > 0) and (answer_word_count <= 10))
        
        # Lexicon ratios
        answer_stopword_ratio, answer_negation_ratio, answer_hedge_ratio = extract_answer_lexicon_ratios(
            answer_text_list[example_index]
        )
        
        # Sentiment features (2) - NEW
        if sentiment_pipeline is not None:
            try:
                question_sent_scores = sentiment_pipeline([question_text_list[example_index]], return_all_scores=True)[0]
                question_sent_dict = {item['label'].lower(): item['score'] for item in question_sent_scores}
                question_sent_negative = float(question_sent_dict.get('negative', 0.0))
                question_sent_positive = float(question_sent_dict.get('positive', 0.0))
                question_sentiment_polarity = question_sent_positive - question_sent_negative
                
                answer_sent_scores = sentiment_pipeline([answer_text_list[example_index]], return_all_scores=True)[0]
                answer_sent_dict = {item['label'].lower(): item['score'] for item in answer_sent_scores}
                answer_sent_negative = float(answer_sent_dict.get('negative', 0.0))
                answer_sent_positive = float(answer_sent_dict.get('positive', 0.0))
                answer_sentiment_polarity = answer_sent_positive - answer_sent_negative
            except Exception:
                question_sentiment_polarity = 0.0
                answer_sentiment_polarity = 0.0
        else:
            question_sentiment_polarity = 0.0
            answer_sentiment_polarity = 0.0
        
        # Structural feature (1) - NEW
        answer_text = answer_text_list[example_index]
        answer_sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip()) if answer_text else []
        answer_sentences = [s for s in answer_sentences if s]
        answer_sentence_count = len(answer_sentences)
        answer_char_count = len(answer_text)
        answer_char_per_sentence = float(answer_char_count / max(1, answer_sentence_count))
        
        # Metadata features (3) - NEW
        def safe_float_convert(value):
            if value is None:
                return 0.0
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        if metadata_dict:
            inaudible_val = metadata_dict.get('inaudible')
            inaudible = safe_float_convert(inaudible_val[example_index] if inaudible_val and example_index < len(inaudible_val) else None)
            
            multiple_questions_val = metadata_dict.get('multiple_questions')
            multiple_questions = safe_float_convert(multiple_questions_val[example_index] if multiple_questions_val and example_index < len(multiple_questions_val) else None)
            
            affirmative_questions_val = metadata_dict.get('affirmative_questions')
            affirmative_questions = safe_float_convert(affirmative_questions_val[example_index] if affirmative_questions_val and example_index < len(affirmative_questions_val) else None)
        else:
            inaudible = 0.0
            multiple_questions = 0.0
            affirmative_questions = 0.0
        
        # Build feature row (25 features)
        feature_rows.append([
            # Model-dependent features (7)
            question_model_token_count,
            answer_model_token_count,
            attention_mass_q_to_a_per_qtoken,
            attention_mass_a_to_q_per_atoken,
            focus_token_to_answer_strength,
            answer_token_to_focus_strength,
            focus_token_coverage_ratio,
            
            # Model-independent features - original (12)
            float(tfidf_cosine_sim[example_index]),
            content_jaccard_q_a,
            question_content_coverage_in_answer,
            answer_content_word_ratio,
            answer_digit_groups_per_word,
            float(refusal_pattern_match_count),
            float(clarification_pattern_match_count),
            float(answer_question_mark_count),
            float(answer_word_count),
            answer_is_short_question,
            answer_negation_ratio,
            answer_hedge_ratio,
            
            # New features (6)
            question_sentiment_polarity,
            answer_sentiment_polarity,
            answer_char_per_sentence,
            inaudible,
            multiple_questions,
            affirmative_questions,
        ])
    
    feature_matrix = np.array(feature_rows, dtype=np.float32)
    
    feature_names = get_feature_names()  # Returns 25 feature names
    
    return feature_matrix, feature_names, tfidf_vectorizer


def featurize_hf_dataset_in_batches_v2(
    dataset,
    tokenizer,
    model,
    device,
    batch_size: int = 8,
    max_sequence_length: int = 512,
    question_key: str = "interview_question",  # QEvasion dataset: 'interview_question' is the original question, 'question' is paraphrased
    answer_key: str = "interview_answer",  # QEvasion dataset uses 'interview_answer', not 'answer'
    show_progress: bool = True,
    tfidf_vectorizer=None,
    model_independent_features: Optional[np.ndarray] = None,
    sentiment_pipeline=None,
    metadata_keys: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, List[str], Any]:
    """
    Extract features from HuggingFace dataset in batches
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: torch device
        batch_size: Batch size for processing
        max_sequence_length: Max sequence length
        question_key: Key for question text in dataset (default: 'interview_question' for QEvasion - original question, not paraphrased 'question')
        answer_key: Key for answer text in dataset (default: 'interview_answer' for QEvasion)
        show_progress: Show progress bar
        tfidf_vectorizer: Pre-fitted TF-IDF vectorizer (or None to fit new on first batch)
        model_independent_features: Optional pre-extracted model-independent features (N, 18). If provided, only model-dependent features are extracted.
        sentiment_pipeline: Optional HuggingFace sentiment pipeline (for sentiment features)
        metadata_keys: Optional dict mapping feature names to dataset keys {'inaudible': 'inaudible', ...}
    
    Returns:
        feature_matrix: (N, 25) numpy array - ALWAYS 25 features
        feature_names: List of 25 feature names
        tfidf_vectorizer: Fitted TF-IDF vectorizer (for reuse)
    
    Note:
        For QEvasion dataset:
        - question_key should be 'interview_question' (original question, NOT 'question' which is paraphrased)
        - answer_key should be 'interview_answer' (not 'answer')
    """
    model.eval()
    
    question_texts = dataset[question_key]
    answer_texts = dataset[answer_key]
    
    n_samples = len(question_texts)
    
    # If model-independent features provided, use efficiency mode (only extract model-dependent)
    if model_independent_features is not None:
        if model_independent_features.shape[0] != n_samples:
            raise ValueError(
                f"Model-independent features shape mismatch: "
                f"expected {n_samples} samples, got {model_independent_features.shape[0]}"
            )
        
        # Extract only model-dependent features
        model_dependent_features, model_dep_names = featurize_model_dependent_features(
            dataset,
            tokenizer,
            model,
            device,
            question_key=question_key,
            answer_key=answer_key,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            show_progress=show_progress,
        )
        
        # Combine: model-dependent (7) + model-independent (18) = 25 features
        # Order: model-dependent first, then model-independent
        feature_matrix = np.hstack([model_dependent_features, model_independent_features])
        feature_names = get_model_dependent_feature_names() + get_model_independent_feature_names()
        
        return feature_matrix, feature_names, None  # tfidf_vectorizer not needed in efficiency mode
    
    # Normal mode: Extract all 25 features
    # Extract metadata if provided
    metadata_dict = None
    if metadata_keys:
        metadata_dict = {}
        for feature_name, dataset_key in metadata_keys.items():
            if dataset_key in dataset.column_names:
                metadata_dict[feature_name] = dataset[dataset_key]
    
    all_features = []
    iterator = range(0, n_samples, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting features")
    
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, n_samples)
        batch_questions = question_texts[start_idx:end_idx]
        batch_answers = answer_texts[start_idx:end_idx]
        
        # Extract metadata for this batch
        batch_metadata = None
        if metadata_dict:
            batch_metadata = {
                key: values[start_idx:end_idx] for key, values in metadata_dict.items()
            }
        
        batch_features, feature_names, tfidf_vectorizer = extract_batch_features_v2(
            batch_questions,
            batch_answers,
            tokenizer,
            model,
            device,
            max_sequence_length=max_sequence_length,
            tfidf_vectorizer=tfidf_vectorizer,
            sentiment_pipeline=sentiment_pipeline,
            metadata_dict=batch_metadata,
        )
        
        all_features.append(batch_features)
    
    feature_matrix = np.vstack(all_features)
    return feature_matrix, feature_names, tfidf_vectorizer


def extract_model_independent_features_batch(
    question_text_list: List[str],
    answer_text_list: List[str],
    tfidf_vectorizer=None,
    sentiment_pipeline=None,
    metadata_dict: Optional[Dict[str, List]] = None,
) -> Tuple[np.ndarray, List[str], Any]:
    """
    Extract model-independent features (sadece text'e bağlı, 1 kez çıkar, tüm modeller için kullan)
    
    Args:
        question_text_list: List of question texts
        answer_text_list: List of answer texts
        tfidf_vectorizer: Pre-fitted TF-IDF vectorizer
        sentiment_pipeline: HuggingFace sentiment pipeline (cardiffnlp/twitter-roberta-base-sentiment-latest)
        metadata_dict: Dict with metadata columns {'inaudible': [...], 'multiple_questions': [...], 'affirmative_questions': [...]}
    
    Returns:
        feature_matrix: (N, 18) numpy array
        feature_names: List of 18 feature names
        tfidf_vectorizer: Fitted TF-IDF vectorizer (for reuse)
    """
    batch_size = len(question_text_list)
    
    # Compute TF-IDF cosine similarity
    tfidf_cosine_sim, tfidf_vectorizer = compute_tfidf_cosine_similarity_per_pair(
        question_text_list,
        answer_text_list,
        fitted_vectorizer=tfidf_vectorizer,
    )
    
    feature_rows = []
    
    for example_index in range(batch_size):
        question_text = question_text_list[example_index]
        answer_text = answer_text_list[example_index]
        
        # ============================================================
        # MEVCUT MODEL-INDEPENDENT FEATURES (12 feature)
        # ============================================================
        
        # Content word metrics
        content_jaccard_q_a, question_content_coverage_in_answer = compute_content_word_set_metrics(
            question_text,
            answer_text,
        )
        
        # Answer content word ratio
        answer_content_word_ratio = float(content_word_ratio(answer_text))
        
        # Digit groups per word
        raw_digit_group_count = digit_group_count(answer_text)
        _, _, _, answer_word_count = extract_pattern_based_features_case_insensitive(answer_text)
        answer_digit_groups_per_word = float(raw_digit_group_count / max(1, answer_word_count))
        
        # Pattern-based features
        refusal_pattern_match_count, clarification_pattern_match_count, answer_question_mark_count, answer_word_count = (
            extract_pattern_based_features_case_insensitive(answer_text)
        )
        
        # Short question style
        answer_is_short_question = float((answer_question_mark_count > 0) and (answer_word_count <= 10))
        
        # Lexicon ratios
        answer_stopword_ratio, answer_negation_ratio, answer_hedge_ratio = extract_answer_lexicon_ratios(
            answer_text
        )
        
        # ============================================================
        # YENİ: SENTIMENT FEATURES (2 feature) - Polarity scores
        # ============================================================
        # Note: Sentiment pipeline should be initialized with deterministic settings
        # for reproducibility. The pipeline itself is deterministic if the model
        # weights are fixed, but batch processing order should be consistent.
        if sentiment_pipeline is not None:
            try:
                # Question sentiment
                # Process one at a time for deterministic results (batch processing may vary)
                question_sent_scores = sentiment_pipeline([question_text], return_all_scores=True)[0]
                question_sent_dict = {item['label'].lower(): item['score'] for item in question_sent_scores}
                question_sent_negative = float(question_sent_dict.get('negative', 0.0))
                question_sent_positive = float(question_sent_dict.get('positive', 0.0))
                question_sentiment_polarity = question_sent_positive - question_sent_negative  # Range: -1.0 to +1.0
                
                # Answer sentiment
                answer_sent_scores = sentiment_pipeline([answer_text], return_all_scores=True)[0]
                answer_sent_dict = {item['label'].lower(): item['score'] for item in answer_sent_scores}
                answer_sent_negative = float(answer_sent_dict.get('negative', 0.0))
                answer_sent_positive = float(answer_sent_dict.get('positive', 0.0))
                answer_sentiment_polarity = answer_sent_positive - answer_sent_negative  # Range: -1.0 to +1.0
            except Exception as e:
                # Fallback: set to 0 if sentiment extraction fails
                question_sentiment_polarity = 0.0
                answer_sentiment_polarity = 0.0
        else:
            question_sentiment_polarity = 0.0
            answer_sentiment_polarity = 0.0
        
        # ============================================================
        # YENİ: STRUCTURAL FEATURES (1 feature)
        # ============================================================
        
        # Answer sentence count and character count
        answer_sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip()) if answer_text else []
        answer_sentences = [s for s in answer_sentences if s]
        answer_sentence_count = len(answer_sentences)
        answer_char_count = len(answer_text)
        
        # Structural feature: answer_char_per_sentence
        answer_char_per_sentence = float(answer_char_count / max(1, answer_sentence_count))
        
        # ============================================================
        # YENİ: METADATA FEATURES (3 feature) - Dataset'ten gelecek
        # ============================================================
        def safe_float_convert(value):
            if value is None:
                return 0.0
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        if metadata_dict:
            inaudible_val = metadata_dict.get('inaudible')
            inaudible = safe_float_convert(inaudible_val[example_index] if inaudible_val and example_index < len(inaudible_val) else None)
            
            multiple_questions_val = metadata_dict.get('multiple_questions')
            multiple_questions = safe_float_convert(multiple_questions_val[example_index] if multiple_questions_val and example_index < len(multiple_questions_val) else None)
            
            affirmative_questions_val = metadata_dict.get('affirmative_questions')
            affirmative_questions = safe_float_convert(affirmative_questions_val[example_index] if affirmative_questions_val and example_index < len(affirmative_questions_val) else None)
        else:
            inaudible = 0.0
            multiple_questions = 0.0
            affirmative_questions = 0.0
        
        # Build feature row (18 feature: 12 mevcut + 2 sentiment + 1 structural + 3 metadata)
        feature_rows.append([
            # TF-IDF (1)
            float(tfidf_cosine_sim[example_index]),
            
            # Content word features (3)
            content_jaccard_q_a,
            question_content_coverage_in_answer,
            answer_content_word_ratio,
            
            # Pattern features (5)
            float(refusal_pattern_match_count),
            float(clarification_pattern_match_count),
            float(answer_question_mark_count),
            float(answer_word_count),
            answer_is_short_question,
            
            # Lexicon ratios (2)
            answer_negation_ratio,
            answer_hedge_ratio,
            
            # Digit groups (1)
            answer_digit_groups_per_word,
            
            # Sentiment features (2) - YENİ
            question_sentiment_polarity,
            answer_sentiment_polarity,
            
            # Structural features (1) - YENİ
            answer_char_per_sentence,
            
            # Metadata features (3) - YENİ
            inaudible,
            multiple_questions,
            affirmative_questions,
        ])
    
    feature_matrix = np.array(feature_rows, dtype=np.float32)
    feature_names = get_model_independent_feature_names()
    
    return feature_matrix, feature_names, tfidf_vectorizer


def featurize_model_independent_features(
    dataset,
    question_key: str = "question",
    answer_key: str = "interview_answer",
    batch_size: int = 32,
    show_progress: bool = True,
    sentiment_pipeline=None,
    metadata_keys: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract model-independent features from dataset (1 kez çıkar, tüm modeller için kullan)
    
    Args:
        dataset: HuggingFace dataset
        question_key: Key for question text (default: 'question' for paraphrased question)
        answer_key: Key for answer text (default: 'interview_answer')
        batch_size: Batch size for processing (larger for sentiment pipeline)
        show_progress: Show progress bar
        sentiment_pipeline: HuggingFace sentiment pipeline (or None to skip sentiment)
        metadata_keys: Dict mapping feature names to dataset keys
                      e.g., {'inaudible': 'inaudible', 'multiple_questions': 'multiple_questions', 'affirmative_questions': 'affirmative_questions'}
    
    Returns:
        feature_matrix: (N, 18) numpy array
        feature_names: List of 18 feature names
    """
    question_texts = dataset[question_key]
    answer_texts = dataset[answer_key]
    
    n_samples = len(question_texts)
    all_features = []
    tfidf_vectorizer = None
    
    # Extract metadata if provided
    metadata_dict = None
    if metadata_keys:
        metadata_dict = {}
        for feature_name, dataset_key in metadata_keys.items():
            if dataset_key in dataset.column_names:
                metadata_dict[feature_name] = dataset[dataset_key]
    
    iterator = range(0, n_samples, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting model-independent features")
    
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, n_samples)
        batch_questions = question_texts[start_idx:end_idx]
        batch_answers = answer_texts[start_idx:end_idx]
        
        # Extract metadata for this batch
        batch_metadata = None
        if metadata_dict:
            batch_metadata = {
                key: values[start_idx:end_idx] for key, values in metadata_dict.items()
            }
        
        batch_features, feature_names, tfidf_vectorizer = extract_model_independent_features_batch(
            batch_questions,
            batch_answers,
            tfidf_vectorizer=tfidf_vectorizer,
            sentiment_pipeline=sentiment_pipeline,
            metadata_dict=batch_metadata,
        )
        
        all_features.append(batch_features)
    
    feature_matrix = np.vstack(all_features)
    return feature_matrix, feature_names


def extract_model_dependent_features_batch(
    question_text_list: List[str],
    answer_text_list: List[str],
    tokenizer,
    model,
    device,
    max_sequence_length: int = 512,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract model-dependent features (model/tokenizer'a bağlı, her model için ayrı çıkar)
    
    Args:
        question_text_list: List of question texts
        answer_text_list: List of answer texts
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model (must support output_attentions=True)
        device: torch device
        max_sequence_length: Max sequence length for tokenization
    
    Returns:
        feature_matrix: (N, 7) numpy array
        feature_names: List of 7 feature names
    """
    # Tokenize
    model_inputs = tokenizer(
        question_text_list,
        answer_text_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_sequence_length,
    )
    
    # Move to device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    # Forward pass (with attentions)
    with torch.no_grad():
        model_outputs = model(**model_inputs, output_attentions=True)
    
    # Last-layer attention averaged over heads: (B, Heads, T, T) -> (B, T, T)
    last_layer_attention = model_outputs.attentions[-1].mean(dim=1)
    
    # Split question/answer segments
    question_position_mask, answer_position_mask = split_segments_safe(
        model_inputs,
        tokenizer,
        debug=False,
        max_print=0,
    )
    
    batch_size, sequence_length, _ = last_layer_attention.shape
    
    feature_rows = []
    
    for example_index in range(batch_size):
        # Token positions
        question_token_positions = question_position_mask[example_index].nonzero(as_tuple=True)[0]
        answer_token_positions = answer_position_mask[example_index].nonzero(as_tuple=True)[0]
        
        # Model token counts
        question_model_token_count = int(len(question_token_positions))
        answer_model_token_count = int(len(answer_token_positions))
        
        # Defaults for attention features
        attention_mass_q_to_a_per_qtoken = 0.0
        attention_mass_a_to_q_per_atoken = 0.0
        focus_token_to_answer_strength = 0.0
        answer_token_to_focus_strength = 0.0
        focus_token_coverage_ratio = 0.0
        
        if question_model_token_count > 0 and answer_model_token_count > 0:
            # Attention submatrices
            q_to_a_attention = last_layer_attention[example_index][question_token_positions][:, answer_token_positions]
            a_to_q_attention = last_layer_attention[example_index][answer_token_positions][:, question_token_positions]
            
            # Normalized attention mass
            attention_mass_q_to_a_per_qtoken = float(q_to_a_attention.sum().item() / max(1, question_model_token_count))
            attention_mass_a_to_q_per_atoken = float(a_to_q_attention.sum().item() / max(1, answer_model_token_count))
            
            # Focus token selection (top-k by centrality)
            question_outgoing_sum = last_layer_attention[example_index][question_token_positions].sum(dim=1)
            question_incoming_sum = last_layer_attention[example_index][:, question_token_positions].sum(dim=0)
            question_token_centrality_score = question_outgoing_sum + question_incoming_sum
            
            max_focus_token_count = min(8, question_model_token_count)
            top_focus_indices_within_question = torch.topk(
                question_token_centrality_score,
                k=max_focus_token_count
            ).indices
            focus_token_positions = question_token_positions[top_focus_indices_within_question]
            
            # Focus -> answer attention
            focus_to_answer_attention = last_layer_attention[example_index][focus_token_positions][:, answer_token_positions]
            answer_to_focus_attention = last_layer_attention[example_index][answer_token_positions][:, focus_token_positions]
            
            # Focus features
            focus_token_to_answer_strength = float(focus_to_answer_attention.max(dim=1).values.mean().item())
            answer_token_to_focus_strength = float(answer_to_focus_attention.max(dim=1).values.mean().item())
            
            # Focus coverage
            attention_threshold = 0.08
            focus_token_coverage_ratio = float(
                (focus_to_answer_attention.max(dim=1).values > attention_threshold).float().mean().item()
            )
        
        # Build feature row (7 model-dependent features)
        feature_rows.append([
            # Token counts (2)
            question_model_token_count,
            answer_model_token_count,
            
            # Attention features (2)
            attention_mass_q_to_a_per_qtoken,
            attention_mass_a_to_q_per_atoken,
            
            # Focus features (3)
            focus_token_to_answer_strength,
            answer_token_to_focus_strength,
            focus_token_coverage_ratio,
        ])
    
    feature_matrix = np.array(feature_rows, dtype=np.float32)
    feature_names = get_model_dependent_feature_names()
    
    return feature_matrix, feature_names


def featurize_model_dependent_features(
    dataset,
    tokenizer,
    model,
    device,
    question_key: str = "question",
    answer_key: str = "interview_answer",
    batch_size: int = 8,
    max_sequence_length: int = 512,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract model-dependent features from dataset (her model için ayrı çıkar)
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: torch device
        question_key: Key for question text in dataset
        answer_key: Key for answer text in dataset
        batch_size: Batch size for processing
        max_sequence_length: Max sequence length
        show_progress: Show progress bar
    
    Returns:
        feature_matrix: (N, 7) numpy array
        feature_names: List of 7 feature names
    """
    model.eval()
    
    question_texts = dataset[question_key]
    answer_texts = dataset[answer_key]
    
    n_samples = len(question_texts)
    all_features = []
    
    iterator = range(0, n_samples, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting model-dependent features")
    
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, n_samples)
        batch_questions = question_texts[start_idx:end_idx]
        batch_answers = answer_texts[start_idx:end_idx]
        
        batch_features, feature_names = extract_model_dependent_features_batch(
            batch_questions,
            batch_answers,
            tokenizer,
            model,
            device,
            max_sequence_length=max_sequence_length,
        )
        
        all_features.append(batch_features)
    
    feature_matrix = np.vstack(all_features)
    return feature_matrix, feature_names

