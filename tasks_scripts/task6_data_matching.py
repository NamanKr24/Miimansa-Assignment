"""
Task 6: Entity Linking for Adverse Drug Reactions (ADRs)

This script performs entity linking, a process of connecting a mentioned entity
in the text to a standardized entry in a knowledge base. Here, we link the ADRs
predicted by our model to their formal SNOMED CT codes and descriptions.

The process involves two main stages:
1.  Data Integration: Combining the 'original' and 'sct' annotation files to
    create a unified data structure that links ground truth text to standard codes.
2.  Matching: Implementing and comparing two different techniques to match our
    model's predicted ADRs against the integrated ground truth data:
    a) Approximate String Matching (lexical similarity).
    b) Embedding Model Matching (semantic similarity).
"""

import os
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer, util

def combine_data_structures(dataset_path, filename):
    """
    Merges data from 'original' and 'sct' files for a single document.

    This function creates a unified dictionary where each entity is enriched with
    its standard code and description from the SNOMED CT (sct) annotations,
    if available.

    Args:
        dataset_path (str): The root path to the CADEC dataset.
        filename (str): The .txt filename of the document to process.

    Returns:
        dict: A dictionary where keys are entity tags and values contain the
              merged information (label, text, standard code, and description).
    """
    combined_data = {}
    
    # Construct full paths to the required annotation files.
    original_path = os.path.join(dataset_path, 'cadec', 'original', filename.replace('.txt','.ann'))
    sct_path = os.path.join(dataset_path, 'cadec', 'sct', filename.replace('.txt','.ann'))

    # 1. Parse the 'original' file to get entity tags, labels, and text.
    original_data = {}
    try:
        with open(original_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    tag, label_part, text = parts[0], parts[1], parts[-1]
                    label = label_part.split(' ')[0]
                    original_data[tag] = {'label': label, 'text': text}
    except FileNotFoundError:
        print(f"Error: Original file not found at {original_path}")
        return {}

    # 2. Parse the 'sct' file to get standard codes and descriptions.
    sct_data = {}
    try:
        with open(sct_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                # SCT format is more complex; we need at least 5 parts for safe parsing.
                if len(parts) >= 5:
                    tag_with_t = parts[0]
                    tag = tag_with_t[1:] # Remove the leading 'T' from the tag to match 'original'.
                    sct_code = parts[1]
                    sct_description = parts[2]
                    sct_data[tag] = {'sct_code': sct_code, 'sct_description': sct_description}
    except FileNotFoundError:
        print(f"Error: SCT file not found at {sct_path}")
        # We can still proceed without SCT data, so we don't return here.

    # 3. Merge the two data sources based on the common entity tag.
    for tag, orig_info in original_data.items():
        if tag in sct_data:
            # If a standard code exists for this entity, add it.
            sct_info = sct_data[tag]
            combined_data[tag] = {
                'label_type': orig_info['label'],
                'ground_truth_text': orig_info['text'],
                'standard_code': sct_info['sct_code'],
                'standard_description': sct_info['sct_description']
            }
        else:
            # If not, fill in with None.
            combined_data[tag] = {
                'label_type': orig_info['label'],
                'ground_truth_text': orig_info['text'],
                'standard_code': None,
                'standard_description': None
            }
    return combined_data

def get_predicted_adr_labels_for_task6(filepath, get_predicted_labels_func):
    """
    Wrapper function to get a clean list of predicted ADR entity texts.
    """
    all_labels = get_predicted_labels_func(filepath)
    # Filter the full output of the NER pipeline to get only ADRs for this task.
    return [label['text'].lower() for label in all_labels if label['label'] == 'ADR']


def approximate_string_match(combined_data, predicted_adr_labels):
    """
    Links predicted ADRs to standard codes using lexical (character-level) similarity.

    This method calculates the character-level similarity between the predicted
    text and the ground truth text. It's effective for catching minor misspellings
    or variations.

    Args:
        combined_data (dict): The merged data structure from `combine_data_structures`.
        predicted_adr_labels (list): A list of ADR texts predicted by the model.
    """
    print("\n--- a) Approximate String Match Results ---")
    
    # Create a lookup dictionary of only the ground truth ADRs that have a standard code.
    adr_ground_truth_data = {
        data['ground_truth_text'].lower(): data
        for data in combined_data.values() if data['label_type'] == 'ADR' and data['standard_code']
    }

    if not adr_ground_truth_data:
        print("No ADR ground truth data with standard codes found for string matching.")
        return

    for predicted_text in predicted_adr_labels:
        best_match_score = 0
        best_match_data = None
        
        # Compare the predicted text against all possible ground truth texts.
        for gt_text, gt_data in adr_ground_truth_data.items():
            # Use fuzzy string matching to get a similarity ratio (0-100).
            score = fuzz.ratio(predicted_text, gt_text)
            if score > best_match_score:
                best_match_score = score
                best_match_data = gt_data
        
        # Use a threshold to filter out poor-quality matches.
        if best_match_data and best_match_score > 70:
            print(f"\nPredicted Text: '{predicted_text}'")
            print(f"  > Best Match (Score: {best_match_score}): '{best_match_data['ground_truth_text']}'")
            print(f"  > Standard Code: {best_match_data['standard_code']}")
            print(f"  > Standard Description: {best_match_data['standard_description']}")
        else:
            print(f"\nPredicted Text: '{predicted_text}'")
            print(f"  > No suitable string match found (highest score: {best_match_score}).")

def embedding_model_match(combined_data, predicted_adr_labels):
    """
    Links predicted ADRs to standard codes using semantic similarity.

    This method converts both the predicted text and ground truth texts into
    numerical vectors (embeddings). It then finds the best match by calculating
    the cosine similarity, which measures the similarity in meaning, not just
    character overlap.

    Args:
        combined_data (dict): The merged data structure.
        predicted_adr_labels (list): A list of ADR texts predicted by the model.
    """
    print("\n--- b) Embedding Model Match Results ---")
    
    try:
        # 'all-MiniLM-L6-v2' is a lightweight but powerful model for semantic similarity.
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        return
    
    # Create a list of the ground truth ADRs that have standard codes.
    adr_ground_truth_list = [
        data for data in combined_data.values() if data['label_type'] == 'ADR' and data['standard_code']
    ]

    if not adr_ground_truth_list:
        print("No ADR ground truth data with standard codes found for embedding matching.")
        return

    # For efficiency, encode all ground truth texts into embeddings in a single batch.
    gt_texts = [item['ground_truth_text'] for item in adr_ground_truth_list]
    gt_embeddings = model.encode(gt_texts, convert_to_tensor=True)
    
    for predicted_text in predicted_adr_labels:
        # Encode the single predicted text.
        predicted_embedding = model.encode(predicted_text, convert_to_tensor=True)
        
        # Efficiently compute cosine similarity between the prediction and all ground truths.
        cosine_scores = util.cos_sim(predicted_embedding, gt_embeddings)[0]
        
        # Find the index of the highest score.
        best_match_index = cosine_scores.argmax()
        best_match_score = cosine_scores[best_match_index].item()
        
        # Use a threshold to ensure semantic relevance.
        if best_match_score > 0.6:
            best_match_data = adr_ground_truth_list[best_match_index]
            print(f"\nPredicted Text: '{predicted_text}'")
            print(f"  > Best Match (Score: {best_match_score:.4f}): '{best_match_data['ground_truth_text']}'")
            print(f"  > Standard Code: {best_match_data['standard_code']}")
            print(f"  > Standard Description: {best_match_data['standard_description']}")
        else:
            print(f"\nPredicted Text: '{predicted_text}'")
            print(f"  > No suitable semantic match found (highest score: {best_match_score:.4f}).")