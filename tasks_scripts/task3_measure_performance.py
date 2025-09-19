"""
Task 3: Performance Measurement Against 'Original' Ground Truth

This script evaluates the performance of the NER model from Task 2. It compares
the model's predicted entities against the ground truth annotations found in the
'original' directory for a single document.

The chosen evaluation metric is a strict, exact-match F1-score, calculated for
each entity category (ADR, Drug, Disease, Symptom). This provides a clear and
objective measure of the model's accuracy and completeness on a given text.
"""

import os
from collections import defaultdict

def get_labels_from_file(filepath):
    """
    Parses a single ground truth .ann file from the 'original' directory.

    Args:
        filepath (str): The path to the .ann file to be parsed.

    Returns:
        defaultdict: A dictionary mapping each label type to a list of its
                     associated entity texts (e.g., {'ADR': ['headache', 'nausea']}).
    """
    labels = defaultdict(list)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    label_part = parts[1]
                    label_type = label_part.split(' ')[0]
                    # Normalize ground truth text to lowercase for case-insensitive comparison.
                    entity_text = parts[-1].lower()
                    
                    if label_type in ['ADR', 'Drug', 'Disease', 'Symptom']:
                        labels[label_type].append(entity_text)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    return labels

def calculate_ner_metrics(predicted_labels, ground_truth_labels):
    """
    Calculates precision, recall, and F1-score based on exact entity matching.

    This function compares the set of predicted entities with the set of ground
    truth entities for each label type to determine the number of true positives,
    false positives, and false negatives. This strict, exact-match approach ensures
    the model is evaluated on its ability to be precise in both classification
    and boundary detection.

    Args:
        predicted_labels (dict): A dictionary of predicted entities from the model.
        ground_truth_labels (dict): A dictionary of ground truth entities from the file.

    Returns:
        dict: A dictionary containing the precision, recall, and F1-score
              for each entity label type.
    """
    metrics = {}
    
    # Ensure we evaluate all label types present in either predictions or ground truth.
    all_label_types = set(predicted_labels.keys()) | set(ground_truth_labels.keys())
    
    for label_type in sorted(list(all_label_types)):
        
        # Convert lists to sets for efficient comparison and to handle duplicates.
        gt_set = set(ground_truth_labels.get(label_type, []))
        pred_set = set(predicted_labels.get(label_type, []))
        
        # --- Calculate Core Metrics ---
        # True Positives: Entities correctly identified by the model (exact match).
        true_positives = len(gt_set.intersection(pred_set))
        # False Positives: Entities the model predicted but are not in the ground truth.
        false_positives = len(pred_set - gt_set)
        # False Negatives: Entities in the ground truth that the model missed.
        false_negatives = len(gt_set - pred_set)
        
        # --- Calculate Precision, Recall, and F1-Score ---
        # Precision: Of all the entities the model predicted, how many were correct?
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        # Recall: Of all the actual entities in the text, how many did the model find?
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        # F1-Score: The harmonic mean of precision and recall, providing a single balanced score.
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label_type] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    return metrics

def measure_performance(dataset_path, filename, get_predicted_labels_func):
    """
    Orchestrates the performance measurement process for a single file.

    It loads the ground truth and the model's predictions, formats them,
    calculates the performance metrics, and prints a formatted report.

    Args:
        dataset_path (str): The root path to the CADEC dataset.
        filename (str): The .ann filename of the file to evaluate.
        get_predicted_labels_func (function): A function that takes a text file
                                             path and returns the model's predictions.
    """
    # Define paths for both the ground truth and the raw text file.
    original_path = os.path.join(dataset_path, 'cadec', 'original', filename)
    text_path = os.path.join(dataset_path, 'cadec', 'text', filename.replace('.ann', '.txt'))
    
    # Step 1: Load the ground truth labels from the .ann file.
    ground_truth_labels = get_labels_from_file(original_path)
    
    # Step 2: Get predictions from the Task 2 NER pipeline by calling the provided function.
    predicted_labels_list = get_predicted_labels_func(text_path)
    
    # Step 3: Convert the prediction list into the same dictionary format as the
    # ground truth for a fair, apples-to-apples comparison.
    predicted_labels_dict = defaultdict(list)
    for item in predicted_labels_list:
        predicted_labels_dict[item['label']].append(item['text'].lower())
        
    # Step 4: Calculate the metrics.
    metrics = calculate_ner_metrics(predicted_labels_dict, ground_truth_labels)
    
    # --- Display Results in a readable format ---
    print(f"Performance for file: {filename}")
    print("-" * 30)
    for label, scores in sorted(metrics.items()):
        print(f"Label: {label}")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall:    {scores['recall']:.4f}")
        print(f"  F1-Score:  {scores['f1_score']:.4f}")
        print()