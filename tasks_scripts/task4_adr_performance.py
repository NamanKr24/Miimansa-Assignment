"""
Task 4: ADR-Specific Performance Measurement

This script provides a more focused evaluation, measuring the model's performance
specifically on the Adverse Drug Reaction (ADR) category.

Unlike Task 3, this evaluation uses the 'meddra' directory as the ground truth.
The 'meddra' annotations are a curated subset of ADRs, providing a different
and potentially cleaner benchmark for this specific entity type.
"""

import os
from collections import defaultdict

def get_adr_labels_from_meddra(filepath):
    """
    Parses a single .ann file from the 'meddra' directory to extract ADR labels.

    Args:
        filepath (str): The path to the MedDRA annotation file.

    Returns:
        list: A list of all ADR entity texts found in the file, converted to lowercase.
    """
    adr_labels = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # The MedDRA files have a tab-separated format like: Tag  Info  Text
                parts = line.strip().split('\t')
                # We need at least 3 parts to safely access the text at index 2.
                if len(parts) >= 3:
                    # The actual entity text is the third element in the split list.
                    adr_labels.append(parts[2].lower())
    except FileNotFoundError:
        print(f"Error: MedDRA file not found at {filepath}")
    return adr_labels

def calculate_adr_metrics(predicted_labels, ground_truth_labels):
    """
    Calculates precision, recall, and F1-score specifically for ADR labels.

    Args:
        predicted_labels (list): A list of ADR entities predicted by the model.
        ground_truth_labels (list): A list of ADR entities from the MedDRA file.

    Returns:
        dict: A dictionary containing the precision, recall, and F1-score.
    """
    # Convert the lists of strings to sets for efficient, exact-match comparison.
    gt_set = set(ground_truth_labels)
    pred_set = set(predicted_labels)

    # Calculate true positives (overlap), false positives, and false negatives.
    true_positives = len(gt_set.intersection(pred_set))
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)
    
    # Standard formulas for Precision, Recall, and F1-score.
    # Handles division-by-zero cases where no predictions are made or no ground truth exists.
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

def measure_adr_performance(dataset_path, filename, get_predicted_adr_labels_func):
    """
    Orchestrates the ADR-specific performance measurement for a single file.

    Args:
        dataset_path (str): The root path to the CADEC dataset.
        filename (str): The .txt filename of the post to evaluate.
        get_predicted_adr_labels_func (function): A function that takes a text
                                                  file path and returns a list
                                                  of predicted ADR entities.
    """
    # Construct paths to the specialized MedDRA ground truth and the raw text file.
    meddra_path = os.path.join(dataset_path, 'cadec', 'meddra', filename.replace('.txt', '.ann'))
    text_path = os.path.join(dataset_path, 'cadec', 'text', filename)
    
    # Step 1: Load the ground truth ADRs from the MedDRA file.
    ground_truth_labels = get_adr_labels_from_meddra(meddra_path)
    
    # Step 2: Get the model's predictions, which should be pre-filtered to only include ADRs.
    predicted_labels = get_predicted_adr_labels_func(text_path)
    
    # Step 3: Calculate the performance metrics.
    metrics = calculate_adr_metrics(predicted_labels, ground_truth_labels)
    
    # --- Display Results ---
    print(f"ADR Performance for file (vs. MedDRA): {filename}")
    print("-" * 40)
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")