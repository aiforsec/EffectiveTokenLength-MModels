"""
benchmark.py

This script demonstrates a benchmarking process for text-to-image retrieval:
  1. Load a CSV file containing image file names and text documents.
  2. Construct full image paths using `get_ordered_paths`.
  3. Process images and texts to create FAISS indices.
  4. Evaluate retrieval metrics (e.g., Recall@1) at different text truncation limits.
  5. Plot and save the results.

Dependencies:
  - CSV file with "image_filename" and "document" columns.
  - Additional libraries: faiss, numpy, pandas, matplotlib, PIL, torch, transformers.
"""

import os
import sys
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adjust path to find local modules if necessary
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processor import (
    get_ordered_paths,
    process_all_images,
    process_all_texts,
    make_faiss_index,
    get_metrics
)
from feature_extractor import FeatureExtractor

def main():

    # -------------------------------------------------------------------------
    # 1. Initialize the Feature Extractor
    # -------------------------------------------------------------------------
    # The `FeatureExtractor` class supports multiple models including:
    # - "clip" (Hugging Face CLIP)
    # - "align" (Hugging Face ALIGN)
    # - "openclip" (OpenCLIP library)
    # - "blip2" (Hugging Face BLIP-2)
    # - "longclip" (Custom LongCLIP model)
    #
    # Notes:
    # - You **don't need to specify `model_name`** unless overriding the default.  
    # - `pretrained` is **only needed** for OpenCLIP.  
    # - `checkpoint_path` is **only required** for LongCLIP.  
    # - If modifying, ensure you provide the correct parameters for the selected model type.  
    #
    # To use a different model, change `model_type`, `model_name`, and `pretrained` accordingly.
    # For LongCLIP, make sure to provide `checkpoint_path` instead of `pretrained`.

    vectorizer = FeatureExtractor(model_type="openclip",pretrained="openai")

    # -------------------------------------------------------------------------
    # 2. Define file paths and model parameters
    # -------------------------------------------------------------------------
    file_path = r"D:/GitHub/Factify/factify2test_final.csv"      # CSV file with "image_filename" and "document"
    image_folder = r"D:/GitHub/Factify/document_images"          # Directory containing images
    save_dir = r"D:/GitHub/EffectiveTokenLength-MModels/faiss_indexes"                # Where to save FAISS indexes
    os.makedirs(save_dir, exist_ok=True)
    result_dir = r"D:/GitHub/EffectiveTokenLength-MModels/results"                    # Where to save outputs
    os.makedirs(result_dir, exist_ok=True)

    # Name for output files (CSV and plot)
    output_name = "recall_1_vs_tokens_openclip_Factify"

    # -------------------------------------------------------------------------
    # 3. Read CSV and prepare image paths & text documents
    # -------------------------------------------------------------------------
    # Build full paths for images based on the CSV's "image_filename" column
    image_paths = get_ordered_paths(image_folder, file_path)

    # Read the CSV to get text from the "document" column
    df = pd.read_csv(file_path, dtype={'document': str})
    if "document" not in df.columns:
        raise ValueError("CSV file must contain a 'document' column for text data.")
    ordered_texts = df["document"].tolist()

    # -------------------------------------------------------------------------
    # 4. Process images and create an image FAISS index
    # -------------------------------------------------------------------------
    print("Processing images and creating FAISS index...")
    image_vectors = process_all_images(image_paths, vectorizer)
    image_db = make_faiss_index(image_vectors)

    # Save the image FAISS index
    image_db_filename = os.path.join(save_dir, "openclip_factify_image.bin")
    faiss.write_index(image_db, image_db_filename)
    print(f"Image FAISS index saved to: {image_db_filename}")

    # -------------------------------------------------------------------------
    # 5. Evaluate text-to-image retrieval at different token truncation limits
    # -------------------------------------------------------------------------
    recalls_1 = []
    token_steps = range(1, 200, 10)  # Example: test from 1 to 190 tokens in steps of 10

    print("Evaluating text-to-image retrieval with varying token limits...")
    for token_limit in token_steps:
        # Truncate text documents to 'token_limit' tokens
        truncated_texts = [vectorizer.truncate_text_by_tokens(doc, token_limit) for doc in ordered_texts]

        # Process truncated texts to create text vectors
        text_vectors = process_all_texts(truncated_texts, vectorizer)
        text_db = make_faiss_index(text_vectors)

        # Save the text FAISS index for each token limit (optional)
        text_db_filename = os.path.join(save_dir, f"openclip_factify_token_{token_limit}.bin")
        faiss.write_index(text_db, text_db_filename)
        print(f"Text FAISS index saved to: {text_db_filename}")

        # Compute metrics (MRR@10, Recall@1, Recall@10, Recall@1000)
        metrics_df = get_metrics(text_db, image_db, k=1000)
        # The second row of the "Value" column is Recall@1 (index 1)
        recall_1_value = metrics_df["Value"].iloc[1]
        recalls_1.append(recall_1_value)

        print(f"Token limit: {token_limit}, Recall@1: {recall_1_value}")

    # -------------------------------------------------------------------------
    # 6. Save and plot the results (Recall@1 vs. token limit)
    # -------------------------------------------------------------------------
    # Save the numeric results
    recall_csv_path = os.path.join(result_dir, f"{output_name}.csv")
    np.savetxt(recall_csv_path, recalls_1, delimiter=",")
    print(f"Recall@1 values saved to: {recall_csv_path}")

    # Plot Recall@1 over token limits
    plt.figure(figsize=(10, 6))
    plt.plot(token_steps, recalls_1, marker='o', linestyle='-', color='blue', label='Recall@1')
    plt.xlabel("Number of Tokens Used in Query")
    plt.ylabel("Recall@1")
    plt.title("Recall@1 vs. Number of Tokens Used")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(result_dir, f"{output_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    print("Benchmark completed successfully.")

if __name__ == "__main__":
    main()
