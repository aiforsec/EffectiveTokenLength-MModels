import os
import requests
import pandas as pd
from urllib.parse import urlparse

def download_images_and_create_csv(input_tsv, output_dir, output_csv):
    """
    Reads a TSV file with 'document' and 'document_image' columns.
    Downloads images from the URLs in 'document_image' into 'output_dir'.
    For each successfully downloaded image, writes a row to 'output_csv'
    containing:
      - document: original text
      - image_filename: local image filename
    """
    # 1. Read the TSV (tab-separated)
    df = pd.read_csv(input_tsv, sep="\t", encoding="utf-8")

    # 2. Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # We'll store rows for which the download succeeds
    successful_rows = []

    # 3. Iterate over each row and try downloading the image
    for idx, row in df.iterrows():
        # Extract the URL and the text
        url = row["document_image"]
        doc_text = row["document"]
        
        # Skip empty or invalid URLs
        if not isinstance(url, str) or not url.strip():
            print(f"Skipping row {idx}: Invalid or empty URL")
            continue
        
        # Extract filename from the URL path
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)  # e.g. "image.jpg"
        if not filename:
            print(f"Skipping row {idx}: No valid filename in URL '{url}'")
            continue
        
        # Build the local path where we'll save the image
        local_path = os.path.join(output_dir, filename)
        
        # Attempt the download
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # Write the content to a file in chunks
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Downloaded: {url} -> {local_path}")
                
                # Record this row as successful, storing (document, image_filename)
                successful_rows.append((doc_text, filename))
            else:
                print(f"Failed to download {url}. HTTP status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

    # 4. Build a new DataFrame of successful rows
    df_success = pd.DataFrame(successful_rows, columns=["document", "image_filename"])

    # 5. Write the cleaned data to a CSV file (comma-separated)
    df_success.to_csv(output_csv, index=False, encoding="utf-8")

    # Summary
    print(f"\nDownload process complete.")
    print(f"Total rows in original TSV: {len(df)}")
    print(f"Successful downloads: {len(successful_rows)}")
    print(f"Cleaned CSV with successful entries: {output_csv}")

def remove_duplicate_images(input_csv, output_csv):
    # Read the TSV file
    df = pd.read_csv(input_csv)

    # Drop duplicates based on the 'document_image' column
    # keep="first" means we keep the first occurrence, drop subsequent duplicates
    df.drop_duplicates(subset=["image_filename"], keep="first", inplace=True)

    # Write the deduplicated data to a new TSV
    df.to_csv(output_csv, encoding="utf-8")
    
if __name__ == "__main__":
    # Example usage
    input_file = "path/to/your/input.tsv"  # Replace with your actual TSV file path
    images_output_folder = "path/to/your/images"  # Replace with your desired output folder for images
    output_file = "path/to/your/factify.csv"  # Replace with your desired output CSV file path
    
    download_images_and_create_csv(
        input_tsv=input_file,
        output_dir=images_output_folder,
        output_csv=output_file
    )

    remove_duplicate_images(output_file, output_file)
    print(f"Deduplicated file saved to: {output_file}")