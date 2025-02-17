#!/usr/bin/env python
"""
benchmark.py

This script benchmarks the text-to-image retrieval performance of a given feature extraction model
by evaluating Recall@1 across varying text token limits. It performs the following steps:
  1. Loads configuration from a YAML file.
  2. Initializes a FeatureExtractor model.
  3. Reads image file paths and text documents.
  4. Processes images to create a FAISS index for image vectors.
  5. Iteratively truncates text documents to different token limits, processes them into text vectors,
     builds corresponding FAISS indices, and computes retrieval metrics.
  6. Saves the numeric Recall@1 results and generates a plot comparing token limits versus Recall@1.

Required modules:
  - os, yaml, faiss, numpy, pandas, matplotlib
  - data_processor (custom module for processing data and computing metrics)
  - feature_extractor (custom module to load and use the feature extraction model)

Usage:
    python benchmark.py
"""

import os
import yaml
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    # 1. Load configuration from config2.yaml
    # -------------------------------------------------------------------------
    with open("config2.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    paths_cfg = config["paths"]
    data_cfg = config["dataset"]
    token_cfg = config["token_steps"]

    # -------------------------------------------------------------------------
    # 2. Initialize the FeatureExtractor with parameters from config
    # -------------------------------------------------------------------------
    vectorizer = FeatureExtractor(
        model_name=model_cfg.get("name", "clip_base"),
        checkpoint=model_cfg.get("checkpoint_path") if "longclip" in model_cfg.get("name", "").lower() else None
    )

    # -------------------------------------------------------------------------
    # 3. Load file paths & create output directories
    # -------------------------------------------------------------------------
    file_path    = paths_cfg["csv_path"]
    image_folder = paths_cfg["image_folder"]
    save_dir     = paths_cfg["faiss_save_dir"]
    result_dir   = paths_cfg["results_dir"]

    # Create directories if they do not exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Construct an output name based on model and dataset names
    output_name = "recalls_vs_tokens_" + model_cfg["name"] + "_" + data_cfg["name"]

    # -------------------------------------------------------------------------
    # 4. Read CSV and prepare image paths + text documents
    # -------------------------------------------------------------------------
    # Get ordered image paths using a helper function
    image_paths = get_ordered_paths(image_folder, file_path)

    # Load CSV containing text documents
    df = pd.read_csv(file_path, dtype={'document': str})
    if "document" not in df.columns:
        raise ValueError("CSV file must contain a 'document' column for text data.")
    ordered_texts = df["document"].tolist()

    # -------------------------------------------------------------------------
    # 5. Process images and create an image FAISS index
    # -------------------------------------------------------------------------
    print("Processing images and creating FAISS index...")
    # Process images to extract their feature vectors
    image_vectors = process_all_images(image_paths, vectorizer)
    # Build a FAISS index from the image vectors for efficient similarity search
    image_db = make_faiss_index(image_vectors)

    # Save the FAISS index for images
    image_db_filename = os.path.join(save_dir, f"{model_cfg['name']}_image.bin")
    faiss.write_index(image_db, image_db_filename)
    print(f"Image FAISS index saved to: {image_db_filename}")

    # -------------------------------------------------------------------------
    # 6. Evaluate text-to-image retrieval at different token truncation limits
    # -------------------------------------------------------------------------
    recalls_1 = []  # To store Recall@1 values for different token limits

    # Determine token limits from the configuration (e.g., from 1 to 200 in steps of 10)
    token_values = range(token_cfg["start"], token_cfg["stop"], token_cfg["step"])

    print("Evaluating text-to-image retrieval with varying token limits...")
    for token_limit in token_values:
        # Truncate each document to 'token_limit' tokens using the vectorizer's helper method
        truncated_texts = [
            vectorizer.truncate_text_by_tokens(doc, token_limit) for doc in ordered_texts
        ]

        # Process the truncated texts to extract their feature vectors
        text_vectors = process_all_texts(truncated_texts, vectorizer)
        # Create a FAISS index from the text vectors
        text_db = make_faiss_index(text_vectors)

        # Optional: Save the FAISS index for the current token limit
        text_db_filename = os.path.join(save_dir, f"{model_cfg['name']}_token_{token_limit}.bin")
        faiss.write_index(text_db, text_db_filename)
        print(f"Text FAISS index saved to: {text_db_filename}")

        # Compute retrieval metrics (e.g., MRR@10, Recall@1, Recall@10, Recall@1000)
        metrics_df = get_metrics(text_db, image_db, k=1000)

        # Assume that the second row in the "Value" column represents Recall@1
        recall_1_value = metrics_df["Value"].iloc[1]
        recalls_1.append(recall_1_value)
        print(f"Token limit: {token_limit}, Recall@1: {recall_1_value}")

    # -------------------------------------------------------------------------
    # 7. Save numeric results and plot
    # -------------------------------------------------------------------------
    # Save the Recall@1 results as a CSV file
    recall_csv_path = os.path.join(result_dir, f"{output_name}.csv")
    np.savetxt(recall_csv_path, recalls_1, delimiter=",")
    print(f"Recall@1 values saved to: {recall_csv_path}")

    # Generate a plot of Recall@1 vs. Number of Tokens used in the query
    plt.figure(figsize=(10, 6))
    plt.plot(token_values, recalls_1, marker='o', linestyle='-', color='blue', label='Recall@1')
    plt.xlabel("Number of Tokens Used in Query")
    plt.ylabel("Recall@1")
    plt.title("Recall@1 vs. Number of Tokens Used")
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    plot_path = os.path.join(result_dir, f"{output_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")
    print("Benchmark completed successfully.")

if __name__ == "__main__":
    main()
