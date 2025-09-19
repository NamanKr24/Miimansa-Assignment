"""
Task 2: Labeling Forum Posts with a Transformer Model

This script defines the core NER (Named Entity Recognition) pipeline. It uses a
pre-trained transformer model from Hugging Face to perform token classification
on the raw text of a patient's forum post.

The process involves several key steps:
1.  Loading a pre-trained biomedical NER model and its tokenizer.
2.  Running inference on the text to get token-level predictions in BIO format
    (e.g., B_problem, I_problem).
3.  Aggregating the token predictions into complete, human-readable entity phrases.
4.  Refining the model's generic labels (e.g., 'problem', 'treatment') into the
    specific categories required by the assignment (ADR, Disease, Symptom, Drug)
    using a custom, rule-based mapping function.
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def get_refined_label(model_label, entity_text):
    """
    Maps the model's raw output labels to the required assignment categories.

    This function acts as a rule-based layer on top of the model. It uses
    keyword matching to differentiate a generic 'problem' into a more specific
    'ADR', 'Disease', or 'Symptom'. This heuristic approach improves the
    relevance of the final output without needing to re-train the model.

    Args:
        model_label (str): The raw label predicted by the model (e.g., 'problem').
        entity_text (str): The text of the entity being classified.

    Returns:
        str or None: The refined label ('ADR', 'Disease', 'Symptom', 'Drug') or
                     None if the label is not one of the target categories.
    """
    # Simple keyword lists for heuristic-based classification.
    # In a real-world project, these would be expanded into much larger,
    # more comprehensive lists, possibly from a medical dictionary.
    ADR_KEYWORDS = [
        'dizzy', 'nausea', 'rash', 'drowsy', 'headache', 'blurred vision', 
        'gastric', 'weird', 'agony', 'dizziness'
    ]
    DISEASE_KEYWORDS = [
        'arthritis', 'cancer', 'diabetes', 'hypertension', 'infection'
    ]

    # --- Mapping Logic ---
    if model_label == 'treatment':
        return 'Drug'
    
    if model_label == 'problem':
        lower_text = entity_text.lower()
        
        # Check for ADRs first, as they can overlap with symptoms and are often more specific.
        if any(keyword in lower_text for keyword in ADR_KEYWORDS):
            return 'ADR'
        
        # Then, check for specific, known diseases.
        if any(keyword in lower_text for keyword in DISEASE_KEYWORDS):
            return 'Disease'
        
        # If a 'problem' is not a known ADR or Disease, default to 'Symptom'.
        return 'Symptom'
        
    # Ignore other labels predicted by the model (e.g., 'person', 'pronoun')
    # to keep the final output clean.
    return None

def label_text_with_llm(text_path):
    """
    Performs end-to-end NER on a given text file.

    Loads a pre-trained model, tokenizes the input text, predicts entities,
    aggregates them from token pieces, and refines the labels to produce
    a clean list of entities in the required format.

    Args:
        text_path (str): The full path to the input .txt file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              predicted entity, e.g., [{'label': 'ADR', 'text': 'headache'}].
    """
    # Note: This implementation assumes the input text will not exceed the
    # model's maximum sequence length (512 tokens). For a production system,
    # this function would be extended to handle long texts by splitting them
    # into overlapping chunks.
    
    try:
        # This model was chosen as it is pre-trained for biomedical NER tasks.
        # It's loaded from a local path to ensure consistency and avoid
        # reliance on an internet connection during runtime.
        model_name = "medical-ner-proj/bert-medical-ner-proj"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return []

    with open(text_path, 'r', encoding='utf-8') as f:
        post_text = f.read()

    inputs = tokenizer(post_text, return_tensors="pt")
    
    # Run model inference. `torch.no_grad()` is a crucial optimization that
    # disables gradient calculations, speeding up the process and reducing memory.
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    id_to_label = model.config.id2label

    # --- Aggregate BIO labels into entity phrases ---
    # This logic reconstructs full entity text from individual token predictions.
    predicted_labels = []
    current_entity_token_ids = []
    current_entity_label = None
    
    for i, token_id in enumerate(inputs["input_ids"][0]):
        token_label_str = id_to_label[predictions[0][i].item()]
        
        # 'B_' prefix (Beginning) marks the start of a new entity.
        if token_label_str.startswith('B_'):
            # If we were already tracking an entity, finalize and save it first.
            if current_entity_token_ids:
                entity_text = tokenizer.decode(current_entity_token_ids)
                predicted_labels.append({'label': current_entity_label, 'text': entity_text})
            
            # Start tracking the new entity.
            current_entity_token_ids = [token_id.item()]
            current_entity_label = token_label_str[2:] # Get label name, e.g., 'problem'
    
        # 'I_' prefix (Inside) marks a token that continues the current entity.
        elif token_label_str.startswith('I_') and current_entity_label == token_label_str[2:]:
            current_entity_token_ids.append(token_id.item())
    
        # 'O' (Outside) label or a new 'B_' tag means the current entity has ended.
        else:
            if current_entity_token_ids:
                # Use the tokenizer's decode method to correctly reconstruct the text.
                entity_text = tokenizer.decode(current_entity_token_ids)
                predicted_labels.append({'label': current_entity_label, 'text': entity_text})
            
            # Reset trackers for the next entity.
            current_entity_token_ids = []
            current_entity_label = None

    # After the loop, save the very last entity if the text ended mid-entity.
    if current_entity_token_ids:
        entity_text = tokenizer.decode(current_entity_token_ids)
        predicted_labels.append({'label': current_entity_label, 'text': entity_text})
    
    # --- Refine raw model labels into final assignment categories ---
    cleaned_labels = []
    for label_info in predicted_labels:
        # Use the helper function to map generic labels to our specific categories.
        refined_label = get_refined_label(label_info['label'], label_info['text'])
        
        # Only include labels that were successfully mapped.
        if refined_label:
            cleaned_labels.append({'label': refined_label, 'text': label_info['text'].strip()})
    
    return cleaned_labels