import faiss
import numpy as np
#import libraries 
import sys
import os

import faiss
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True


def split_string_into_n_parts(text, n_parts):
    """
    Split a string into `n_parts` roughly equal parts based on word count.

    Args:
        text (str): The input string to be split.
        n_parts (int): The number of parts to split the string into.

    Returns:
        list: A list of strings, each representing a portion of the original text.

    Notes:
        - The splitting is performed by dividing the text into groups of words.
        - If the number of words is not evenly divisible by `n_parts`, 
          the last part may contain more or fewer words than the others.
        - A minimum of one word is ensured in each part.

    """
    words = text.split()
    split_size = max(1, len(words) // n_parts)
    return [" ".join(words[i:i+split_size]) for i in range(0, len(words), split_size)]

def process_text(content, text_vectorizer):
    """
    Process a single text file and extract its vector representation.

    Args:
        path (str): The file path of the text file to process.
        text_vectorizer: An instance of a text vectorizer with a `get_text_features` method.

    Returns:
        np.ndarray: The vector representation of the text content.
    """

    """
    Process text files in the specified paths, vectorize their content, and handle 
    errors gracefully if vectorization fails. If direct vectorization fails, the 
    content is split into progressively smaller parts, and their vectors are averaged 
    to produce the final vector representation.

    Steps:
    1. Open each file and read its content.
    2. Try to vectorize the content directly.
    3. If direct vectorization fails, split the content into multiple parts and 
    attempt vectorization for each part, averaging the resulting vectors.
    4. Print the resulting vector representation for each file.

    """
    
    try:
        # Try to vectorize the content directly
        return text_vectorizer.get_text_features(content)[0]
    except Exception as e:
        
        # Start splitting and retrying
        n_parts = 2
        while True:
            try:
                # Split the content into n_parts and average their vectors
                split_contents = split_string_into_n_parts(str(content), n_parts)
                split_vectors = [text_vectorizer.get_text_features(part)[0] for part in split_contents]
                return np.mean(split_vectors, axis=0)  # Take the average vector
            except Exception as split_error:
                
                n_parts += 1  # Increase the number of parts to split into

def process_all_text(texts, text_vectorizer):
    """
    Process multiple text files and extract vector representations for each.

    Args:
        paths (list of str): List of file paths to process.
        text_vectorizer: An instance of a text vectorizer with a `get_text_features` method.

    Returns:
        list of np.ndarray: List of vector representations for the text files.
        list of str: List of processed file paths.
    """
    all_text_vectors = []
    for content in texts:
        
        text_vector = process_text(content, text_vectorizer)
        all_text_vectors.append(text_vector)

    return all_text_vectors


def process_image(path, image_vectorizer):
    """
    Process a single image file and extract its vector representation.

    Args:
        path (str): The file path of the image to process.
        image_vectorizer: An instance of an image vectorizer with a `get_image_features` method.

    Returns:
        np.ndarray: The vector representation of the image.
    """

    # Open the image
    try:
        image = Image.open(path)  # Ensure the image is in RGB format
    except Exception as e:
        raise ValueError(f"Error loading image at {path}: {e}")

    # Extract the vector using the vectorizer
    return image_vectorizer.get_image_features(image)[0]

def process_all_images(paths, image_vectorizer):
    """
    Process multiple image files and extract vector representations for each.

    Args:
        paths (list of str): List of image file paths to process.
        image_vectorizer: An instance of an image vectorizer with a `get_image_features` method.

    Returns:
        list of np.ndarray: List of vector representations for the image files.
        list of str: List of processed image file paths.
    """
    all_image_vectors = []
    all_image_file_paths = []

    for path in paths:
        all_image_file_paths.append(path)
        
        try:
            # Vectorize the image
            image_vector = process_image(path, image_vectorizer)
            all_image_vectors.append(image_vector)
        except Exception as e:
            
            all_image_vectors.append(np.random.rand(*np.array(all_image_vectors[0]).shape).astype(np.float32))
            print(f"Failed to process image at {path}: {e}")

    return all_image_vectors, all_image_file_paths


def make_faiss_index(vectors):
    """
    Creates a FAISS index for fast similarity search using L2 distance.
    
    Args:
        vectors (list or numpy.ndarray): An array of feature vectors to index.
    
    Returns:
        faiss.IndexFlatL2: A FAISS index containing the provided vectors.
    
    Raises:
        ValueError: If the input vector list is empty.
    """
    if not vectors:
        raise ValueError("Vector array is empty. Provide valid vectors.")
    
    # Convert list of vectors to a NumPy array with float32 type
    vectors = np.array(vectors, dtype=np.float32)
    
    # Determine the dimensionality of the vectors
    dim = vectors.shape[1]
    
    # Create a FAISS index with L2 distance metric
    index = faiss.IndexFlatL2(dim)
    
    # Add vectors to the FAISS index
    index.add(vectors)
    
    return index

   