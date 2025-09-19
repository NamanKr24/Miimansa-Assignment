"""
Task 1: Enumerate and Count Distinct Entities

This script processes the CADEC dataset to identify all unique entities
(ADR, Drug, Disease, Symptom) mentioned across all 'original' annotation files.
It then prints a comprehensive list of these unique entities and their total counts
for each category, providing a baseline understanding of the dataset's contents.
"""

import os
from collections import defaultdict

def enumerate_distinct_entities(dataset_path):
    """
    Parses all .ann files in the 'original' directory to find and count
    all unique entities for each of the four specified label types.

    Args:
        dataset_path (str): The root path to the unzipped CADEC dataset directory.
    """
    # Use a defaultdict of sets to store entities. The `set` data structure
    # automatically handles uniqueness, so we don't need to check for duplicates.
    entities = defaultdict(set)
    label_counts = defaultdict(int)

    # Construct the path to the directory containing the ground truth annotations.
    original_path = os.path.join(dataset_path, 'cadec', 'original')
    
    # Robustly handle cases where the dataset path might be incorrect.
    try:
        filenames = os.listdir(original_path)
    except FileNotFoundError:
        print(f"Error: The directory {original_path} was not found. Please ensure the CADEC dataset is correctly unzipped.")
        return

    # Process each annotation file in the directory.
    for filename in filenames:
        if filename.endswith('.ann'):
            filepath = os.path.join(original_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    # Per the assignment instructions, comment lines should be ignored.
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    # Ensure the line has the expected format (Tag, Label Info, Text) before processing.
                    if len(parts) >= 3:
                        label_part = parts[1]
                        label_type = label_part.split(' ')[0]
                        
                        # Normalize the entity text to lowercase for case-insensitive counting.
                        # This ensures "Pain" and "pain" are treated as the same entity.
                        entity_text = parts[-1].lower()
                        
                        # We only care about the four specified entity types.
                        if label_type in ['ADR', 'Drug', 'Disease', 'Symptom']:
                            entities[label_type].add(entity_text)

    # After collecting all unique entities, get the final counts.
    for label, entity_set in entities.items():
        label_counts[label] = len(entity_set)
    
    # --- Display Results ---
    
    print("Distinct Entities for each label type:")
    # Sort the items for consistent, alphabetical output.
    for label, entity_set in sorted(entities.items()):
        formatted_entities = ", ".join(sorted(list(entity_set)))
        print(f"\n--- {label} ({len(entity_set)}) ---")
        print(f"{formatted_entities}")

    print("\n\nTotal number of distinct entities for each label type:")
    for label, count in sorted(label_counts.items()):
        print(f"- {label}: {count}")