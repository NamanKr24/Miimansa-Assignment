# Named Entity Recognition (NER) for Medical Forum Posts
#### Miimansa Programming Assignment Submission

## 📝 Overview

This project implements a complete, end-to-end Natural Language Processing (NLP) pipeline to extract and analyze medical entities from patient forum posts. The system is built using Python and the Hugging Face `transformers` library, as per the assignment requirements.

The primary goal is to process raw text from the CADEC dataset to identify four key entity types: **Adverse Drug Reactions (ADR)**, **Drugs**, **Diseases**, and **Symptoms**. The project covers the entire lifecycle, from initial data exploration and model inference to performance evaluation and advanced entity linking.

## ✨ Key Features

* **Transformer-Based NER:** Utilizes a pre-trained biomedical NER model for high-quality entity prediction.
* **Robust Pipeline for Long Texts:** The core prediction engine is architected to handle documents of any length by automatically processing text in overlapping chunks (sliding window), overcoming the 512-token limit of standard transformer models.
* **Rule-Based Label Refinement:** Implements a custom mapping layer to intelligently convert the model's generic output labels (e.g., `problem`, `treatment`) into the specific, required categories (e.g., `ADR`, `Disease`).
* **Comprehensive Performance Evaluation:** Measures model performance using Precision, Recall, and F1-score. The final evaluation is a robust, micro-averaged score calculated across a random sample of 50 documents.
* **Dual-Method Entity Linking:** Implements and compares two distinct techniques for linking predicted ADRs to a knowledge base (SNOMED CT codes):
    1.  **Lexical Similarity:** Using approximate string matching.
    2.  **Semantic Similarity:** Using a state-of-the-art sentence transformer model to find matches based on meaning.

## 🚀 How to Run

#### **Prerequisites**
* Python 3.x
* Jupyter Notebook or JupyterLab
* Required packages can be installed via `pip install transformers torch sentence-transformers thefuzz`.

#### **Instructions**
1.  **Dataset:** Download the `CADEC.v2.zip` file and unzip it. The notebook expects the resulting `cadec` directory to be accessible.
2.  **Folder Structure:** Place the six Python scripts (`task1_...py`, `task2_...py`, etc.) in a sub-folder named `tasks_scripts`. The `main_demo.ipynb` notebook should be in the parent directory.
3.  **Execution:** Open and run the `main_demo.ipynb` notebook from top to bottom. The submission is designed to be a Jupyter notebook, as requested, and includes all outputs for each task.

## 📂 Code Structure

The project is modularized into separate scripts for each task, which are then imported and demonstrated in the main notebook.

* **`task1_enumerate_entities.py`**: Scans the entire dataset to enumerate and count all unique entities.
* **`task2_llm_labeling.py`**: Contains the core NER prediction pipeline, including the model loading, text chunking, and label refinement logic.
* **`task3_measure_performance.py`**: Evaluates the NER pipeline against the `original` ground truth annotations.
* **`task4_adr_performance.py`**: Provides a focused evaluation for the `ADR` category against the curated `meddra` ground truth.
* **`task5_random_performance.py`**: Scales the evaluation to run on a random sample of 50 files and calculates overall performance metrics.
* **`task6_data_matching.py`**: Implements the final entity linking task with two different matching algorithms.
* **`main_demo.ipynb`**: The main notebook used to execute and demonstrate the results of each task sequentially.

## 📊 Results & Analysis

The final performance of the pipeline was evaluated across a random sample of 50 files (Task 5).

| Label | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Drug | 0.3226 | 0.6818 | 0.4380 |
| Disease | 0.3750 | 0.2143 | 0.2727 |
| ADR | 0.5000 | 0.0117 | 0.0229 |
| Symptom | 0.0000 | 0.0000 | 0.0000 |

#### **Analysis**
The results indicate the model is most effective at identifying **Drugs**, achieving a strong **Recall** of 68.18%. The performance on **ADR** and **Disease** is lower, reflecting the limitations of the heuristic, keyword-based mapping layer and the strictness of the exact-match evaluation metric. The extremely low recall for **ADR** (1.17%) highlights the challenge of identifying the vast and informal language patients use to describe side effects using a small keyword set.

## 🚧 Challenges, Limitations, and Future Improvements

* **Challenge: Handling Long Text Sequences**
    * **Problem:** The transformer model's 512-token limit caused errors on longer forum posts.
    * **Solution:** The prediction pipeline was re-architected to process text in overlapping chunks, making the system robust to inputs of any length.

* **Limitation: Heuristic Label Mapping**
    * The current keyword-based mapping is a simple heuristic and is the primary factor limiting the system's performance. My lack of a formal medical background restricted the scope and accuracy of these keyword lists.
    * **Future Improvement:** Performance could be significantly improved by consulting a domain expert or using a comprehensive medical dictionary (gazetteer) to expand the keyword lists.

* **Limitation: Reliance on a Pre-trained Model**
    * This project uses a general biomedical model without re-training, as hardware constraints prevented the ideal next step of fine-tuning.
    * **Future Improvement:** With access to appropriate GPU resources, **fine-tuning** the model on the CADEC dataset would be the definitive next step. This would teach the model the specific nuances and labels of this dataset, likely yielding a substantial performance increase and removing the need for the heuristic mapping layer.
