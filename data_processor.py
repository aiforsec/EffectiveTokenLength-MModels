"""
Combined module for image and text vectorization, evaluation, and FAISS indexing.

This module provides functionality to:
  - Construct full image file paths from a CSV containing file names.
  - Process images and texts using provided vectorizer instances.
  - Create a FAISS index for fast similarity search.
  - Evaluate text-to-image retrieval using FAISS indices with metrics such as MRR@10 and Recall@X.
  
Notes:
  - For images, the CSV should have an "image_filename" column.
  - For texts, simply use Pandas' read_csv to extract documents.
  - The vectorizer objects must implement:
      - get_image_features(image) for image vectorization.
      - get_text_features(text) for text vectorization.
"""

import os
from typing import List, Any

import numpy as np
import pandas as pd
import faiss
from PIL import Image, ImageFile

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================
# CSV Processing Functions
# ============================

def get_ordered_paths(directory_path: str, csv_file_path: str) -> List[str]:
    """
    Reads a CSV file containing file names and constructs full paths by joining
    them with the provided directory. Assumes the CSV has an "image_filename" column.

    Args:
        directory_path (str): The base directory where image files are stored.
        csv_file_path (str): Path to the CSV file containing image file names.

    Returns:
        List[str]: List of full file paths in the order provided by the CSV.
    """
    df = pd.read_csv(csv_file_path, dtype={'image_filename': str})
    file_names = df['image_filename'].tolist()
    return [os.path.join(directory_path, file_name) for file_name in file_names]

# ============================
# Utility Functions
# ============================

def split_string_into_n_parts(text: str, n_parts: int) -> List[str]:
    """
    Splits a string into n_parts roughly equal parts based on word count.

    Args:
        text (str): The input string to split.
        n_parts (int): The number of parts to split the text into.

    Returns:
        List[str]: A list of text segments.
    """
    words = text.split()
    split_size = max(1, len(words) // n_parts)
    return [" ".join(words[i:i + split_size]) for i in range(0, len(words), split_size)]

# ============================
# Text Processing Functions
# ============================

def process_text(content: str, text_vectorizer: Any) -> np.ndarray:
    """
    Processes a single text document to extract its vector representation.
    If direct vectorization fails, splits the text into parts and averages their vectors.

    Args:
        content (str): The text content to process.
        text_vectorizer: An object with a `get_text_features` method.

    Returns:
        np.ndarray: The vector representation of the text.
    """
    try:
        # Direct vectorization
        return text_vectorizer.get_text_features(content)[0]
    except Exception:
        # On failure, try splitting the text into parts and averaging the resulting vectors.
        n_parts = 2
        while True:
            try:
                parts = split_string_into_n_parts(content, n_parts)
                vectors = [text_vectorizer.get_text_features(part)[0] for part in parts]
                return np.mean(vectors, axis=0)
            except Exception:
                n_parts += 1  # Increase number of splits until successful

def process_all_texts(texts: List[str], text_vectorizer: Any) -> List[np.ndarray]:
    """
    Processes multiple text documents and extracts their vector representations.

    Args:
        texts (List[str]): A list of text documents.
        text_vectorizer: An object with a `get_text_features` method.

    Returns:
        List[np.ndarray]: A list of vector representations for each text document.
    """
    return [process_text(text, text_vectorizer) for text in texts]

# ============================
# Image Processing Functions
# ============================

def process_image(path: str, image_vectorizer: Any) -> np.ndarray:
    """
    Processes a single image file to extract its vector representation.

    Args:
        path (str): The full path to the image file.
        image_vectorizer: An object with a `get_image_features` method.

    Returns:
        np.ndarray: The vector representation of the image.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    try:
        image = Image.open(path)
    except Exception as e:
        raise ValueError(f"Error loading image at {path}: {e}")
    return image_vectorizer.get_image_features(image)[0]

def process_all_images(paths: List[str], image_vectorizer: Any) -> List[np.ndarray]:
    """
    Processes multiple image files and extracts their vector representations.

    Args:
        paths (List[str]): List of image file paths.
        image_vectorizer: An object with a `get_image_features` method.

    Returns:
        List[np.ndarray]: A list of vector representations for the images.
    """
    vectors = []
    for path in paths:
        try:
            vec = process_image(path, image_vectorizer)
            vectors.append(vec)
        except Exception as e:
            # If processing fails, use a random fallback vector.
            if vectors:
                fallback_shape = np.array(vectors[0]).shape
            else:
                fallback_shape = (512,)  # Adjust fallback dimensions as needed.
            random_vec = np.random.rand(*fallback_shape).astype(np.float32)
            vectors.append(random_vec)
            print(f"Failed to process image at {path}: {e}")
    return vectors

# ============================
# FAISS Index Creation
# ============================

def make_faiss_index(vectors: List[np.ndarray]) -> faiss.IndexFlatL2:
    """
    Creates a FAISS index from a list of feature vectors for fast similarity search using L2 distance.

    Args:
        vectors (List[np.ndarray]): List of feature vectors.

    Returns:
        faiss.IndexFlatL2: The FAISS index built from the provided vectors.

    Raises:
        ValueError: If the vector list is empty.
    """
    if not vectors:
        raise ValueError("Vector array is empty. Provide valid vectors.")
    
    vectors_array = np.array(vectors, dtype=np.float32)
    dim = vectors_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors_array)
    return index

# ============================
# Evaluation Metrics
# ============================

def get_metrics(text_db: faiss.Index, image_db: faiss.Index, k: int = 1000) -> pd.DataFrame:
    """
    Compute evaluation metrics for a text-to-image database search task.
    
    For each text vector in text_db, the function retrieves the top k nearest neighbors
    from image_db and computes:
      - MRR@10: Reciprocal rank of the first correct match among the top 10 results.
      - Recall@1, Recall@10, Recall@1000: Whether the correct match is found in the top 1, 10, or 100 results.
    
    Args:
        text_db (faiss.Index): A FAISS index containing text vectors.
        image_db (faiss.Index): A FAISS index containing image vectors.
        k (int): The number of nearest neighbors to retrieve during the search.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated metrics (MRR@10, Recall@1, Recall@10, Recall@1000).
    """
    metrics = []

    for i in range(text_db.ntotal):
        # Query the text vector
        query = np.array([text_db.reconstruct(i)])
        # Perform the search in the image database
        distances, indices = image_db.search(query, k)
        indices = indices[0]
        metric_row = []

        # Compute MRR@10: reciprocal rank if the correct image (assumed same index as text) is in top 10
        if i in indices[:10]:
            metric_row.append(1 / (np.where(indices[:10] == i)[0][0] + 1))
        else:
            metric_row.append(0)

        # Compute Recall@1, Recall@10, Recall@1000
        for recall in [1, 10, 100]:
            metric_row.append(int(i in indices[:recall]))
        
        metrics.append(metric_row)

    # Calculate the average metrics across all queries
    average_metrics = np.mean(metrics, axis=0)

    # Create a DataFrame to store the results
    metrics_df = pd.DataFrame({
        "Metric": ["MRR@10", "Recall@1", "Recall@10", "Recall@1000"],
        "Value": average_metrics
    })

    return metrics_df