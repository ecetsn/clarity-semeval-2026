"""
Main feature extraction functions
"""
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Any
from .utils import (
    split_segments_safe,
    compute_tfidf_cosine_similarity_per_pair,
    compute_content_word_set_metrics,
    content_word_ratio,
    digit_group_count,
    extract_pattern_based_features_case_insensitive,
    extract_answer_lexicon_ratios
)


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
) -> Tuple[np.ndarray, List[str], Any]:
    """
    Extract 19 Context Tree features from a batch of (question, answer) pairs
    
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
    
    Returns:
        feature_matrix: (N, 19) numpy array
        feature_names: List of 19 feature names
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
        
        # Build feature row (19 features)
        feature_rows.append([
            # Length features (2)
            question_model_token_count,
            answer_model_token_count,
            
            # Attention features (2)
            attention_mass_q_to_a_per_qtoken,
            attention_mass_a_to_q_per_atoken,
            
            # Focus features (3)
            focus_token_to_answer_strength,
            answer_token_to_focus_strength,
            focus_token_coverage_ratio,
            
            # Alignment features (3)
            float(tfidf_cosine_sim[example_index]),
            content_jaccard_q_a,
            question_content_coverage_in_answer,
            
            # Answer surface features (2)
            answer_content_word_ratio,
            answer_digit_groups_per_word,
            
            # Pattern features (5)
            float(refusal_pattern_match_count),
            float(clarification_pattern_match_count),
            float(answer_question_mark_count),
            float(answer_word_count),
            answer_is_short_question,
            
            # Lexicon ratios (2)
            answer_negation_ratio,
            answer_hedge_ratio,
        ])
    
    feature_matrix = np.array(feature_rows, dtype=np.float32)
    
    feature_names = [
        "question_model_token_count",
        "answer_model_token_count",
        "attention_mass_q_to_a_per_qtoken",
        "attention_mass_a_to_q_per_atoken",
        "focus_token_to_answer_strength",
        "answer_token_to_focus_strength",
        "focus_token_coverage_ratio",
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
    ]
    
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
    
    Returns:
        feature_matrix: (N, 19) numpy array
        feature_names: List of 19 feature names
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
    all_features = []
    # Use provided tfidf_vectorizer or start with None (will be fitted on first batch)
    
    iterator = range(0, n_samples, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting features")
    
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, n_samples)
        batch_questions = question_texts[start_idx:end_idx]
        batch_answers = answer_texts[start_idx:end_idx]
        
        batch_features, feature_names, tfidf_vectorizer = extract_batch_features_v2(
            batch_questions,
            batch_answers,
            tokenizer,
            model,
            device,
            max_sequence_length=max_sequence_length,
            tfidf_vectorizer=tfidf_vectorizer,
        )
        
        all_features.append(batch_features)
    
    feature_matrix = np.vstack(all_features)
    return feature_matrix, feature_names, tfidf_vectorizer

