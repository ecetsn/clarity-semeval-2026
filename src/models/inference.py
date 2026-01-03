"""
Inference utilities: Predict from question-answer pairs
"""
import numpy as np
from typing import List, Tuple, Union, Any, Optional
import torch
from pathlib import Path

from ..features.extraction import extract_batch_features_v2, featurize_hf_dataset_in_batches_v2


def predict_from_qa_pairs(
    question: Union[str, List[str]],
    answer: Union[str, List[str]],
    model,
    tokenizer,
    classifier: Any,
    device: torch.device,
    tfidf_vectorizer=None,
    max_sequence_length: int = 512,
    return_proba: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict labels from question-answer pairs using trained classifier.
    
    This function:
    1. Extracts 19 Context Tree features from question-answer pairs
    2. Predicts using trained classifier (classifier already contains StandardScaler if needed)
    
    Args:
        question: Question text(s) - single string or list of strings
        answer: Answer text(s) - single string or list of strings
        model: HuggingFace transformer model (for feature extraction)
        tokenizer: HuggingFace tokenizer
        classifier: Trained sklearn classifier (Pipeline with StandardScaler or standalone)
                    Note: Most classifiers (LogisticRegression, LinearSVC, MLP) already contain
                    StandardScaler in their Pipeline, so no separate scaling is needed.
        device: torch device (cpu or cuda)
        tfidf_vectorizer: Pre-fitted TF-IDF vectorizer (REQUIRED - must be fitted on train set)
                         This is model-independent but task-dependent.
        max_sequence_length: Max sequence length for tokenization
        return_proba: If True, also return probability distributions
    
    Returns:
        predictions: (N,) array of predicted labels
        probabilities: (N, num_classes) array (only if return_proba=True)
    
    Example:
        >>> question = "What is your opinion on climate change?"
        >>> answer = "I think it's an important issue that needs attention."
        >>> pred = predict_from_qa_pairs(question, answer, model, tokenizer, clf, device, tfidf_vectorizer)
        >>> print(pred)  # ['Clear Reply']
    
    Note:
        - Feature extraction happens every time (no caching by default)
        - For batch processing, use predict_batch_from_dataset() for better efficiency
        - TF-IDF vectorizer must be fitted on train set to avoid data leakage
    """
    # Convert single strings to lists for batch processing
    if isinstance(question, str):
        question = [question]
    if isinstance(answer, str):
        answer = [answer]
    
    # Ensure same length
    if len(question) != len(answer):
        raise ValueError(f"Question and answer lists must have same length. Got {len(question)} questions and {len(answer)} answers.")
    
    # Extract features from question-answer pairs
    # This includes: attention features, TF-IDF, content word metrics, pattern features, etc.
    features, feature_names, _ = extract_batch_features_v2(
        question_text_list=question,
        answer_text_list=answer,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_sequence_length=max_sequence_length,
        tfidf_vectorizer=tfidf_vectorizer,  # Must be pre-fitted on train set
    )
    
    # Predict using trained classifier
    # Note: Classifier already contains StandardScaler if it's a Pipeline (LogisticRegression, LinearSVC, MLP)
    # For RandomForest, XGBoost, LightGBM: no scaling needed
    predictions = classifier.predict(features)
    
    if return_proba:
        # Get probability distributions (if classifier supports it)
        try:
            probabilities = classifier.predict_proba(features)
        except AttributeError:
            raise ValueError("Classifier does not support predict_proba(). Set return_proba=False.")
        return predictions, probabilities
    
    return predictions


def predict_batch_from_dataset(
    dataset: List[dict],
    model,
    tokenizer,
    classifier: Any,
    device: torch.device,
    storage_manager=None,
    model_name: str = None,
    task: str = None,
    question_key: str = "interview_question",
    answer_key: str = "interview_answer",
    tfidf_vectorizer=None,
    max_sequence_length: int = 512,
    batch_size: int = 8,
    return_proba: bool = False,
    use_cache: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict labels from a dataset (list of dicts) containing question-answer pairs.
    
    This is a convenience function for batch prediction on datasets.
    Processes data in batches for memory efficiency.
    Supports checkpointing: if features are already extracted and saved, loads them instead.
    
    Args:
        dataset: List of dicts, each containing question_key and answer_key
        model: HuggingFace transformer model
        tokenizer: HuggingFace tokenizer
        classifier: Trained sklearn classifier (already contains StandardScaler if needed)
        device: torch device
        storage_manager: StorageManager instance (for checkpointing features)
        model_name: Model name (e.g., 'bert', 'roberta') - required for checkpointing
        task: Task name (e.g., 'clarity', 'evasion') - required for checkpointing
        question_key: Key for question text in dataset dicts
        answer_key: Key for answer text in dataset dicts
        tfidf_vectorizer: Pre-fitted TF-IDF vectorizer (REQUIRED - must be fitted on train set)
                         
                         IMPORTANT: TF-IDF is TASK-DEPENDENT but MODEL-INDEPENDENT.
                         - Same TF-IDF vectorizer can be used for all models for the same task
                         - However, in current implementation, each model fits its own TF-IDF
                           (this is redundant but maintained for consistency with existing code)
                         - For optimal efficiency, use one TF-IDF per task, not per model
        max_sequence_length: Max sequence length
        batch_size: Batch size for feature extraction (larger = faster but more memory)
        return_proba: If True, also return probability distributions
        use_cache: If True, check Drive for existing features before extraction (checkpoint)
    
    Returns:
        predictions: (N,) array of predicted labels
        probabilities: (N, num_classes) array (only if return_proba=True)
    
    Note:
        - Feature extraction happens in batches for memory efficiency
        - If use_cache=True and storage_manager/model_name/task provided, checks Drive for existing features
        - If features exist in Drive, loads them (no re-extraction)
        - If features don't exist, extracts and saves to Drive for future use
        - Processing time depends on dataset size and batch_size
        - TF-IDF vectorizer is task-dependent but model-independent (same for all models per task)
    """
    # Check if features already exist in Drive (checkpoint)
    features = None
    if use_cache and storage_manager is not None and model_name is not None and task is not None:
        try:
            # Try to load test features from Drive
            features = storage_manager.load_features(model_name, task, 'test')
            print(f"✓ Loaded features from Drive: {model_name}_{task}_test ({features.shape[0]} samples)")
            print(f"  Path: features/raw/X_test_{model_name}_{task}.npy")
        except FileNotFoundError:
            print(f"  Features not found in Drive, extracting for {model_name}_{task}...")
    
    # Extract features if not cached
    if features is None:
        # Extract questions and answers from dataset
        questions = [item[question_key] for item in dataset]
        answers = [item[answer_key] for item in dataset]
        
        # Extract features in batches using extract_batch_features_v2
        model.eval()
        all_features = []
        feature_names = None
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_answers = answers[i:i+batch_size]
            
            batch_features, feature_names, tfidf_vectorizer = extract_batch_features_v2(
                question_text_list=batch_questions,
                answer_text_list=batch_answers,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_sequence_length=max_sequence_length,
                tfidf_vectorizer=tfidf_vectorizer,  # Reuse TF-IDF across batches
            )
            all_features.append(batch_features)
        
        features = np.vstack(all_features)
        
        # Save features to Drive for future use (checkpoint)
        if use_cache and storage_manager is not None and model_name is not None and task is not None:
            storage_manager.save_features(features, model_name, task, 'test', feature_names)
            print(f"✓ Saved features to Drive: {model_name}_{task}_test")
            print(f"  Path: features/raw/X_test_{model_name}_{task}.npy")
            print(f"  Next time, features will be loaded from Drive (no re-extraction needed)")
    
    # Predict using trained classifier
    predictions = classifier.predict(features)
    
    if return_proba:
        try:
            probabilities = classifier.predict_proba(features)
            return predictions, probabilities
        except AttributeError:
            raise ValueError("Classifier does not support predict_proba(). Set return_proba=False.")
    
    return predictions

