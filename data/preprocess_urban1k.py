import os
import glob
import csv

def preprocess_urban1k(caption_dir, image_dir, output_csv):
    """
    Create a CSV file with two columns:
      - document: full text content from each caption (.txt) file
      - image_filename: corresponding image filename (e.g., '1.jpg')
    """
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Collect all .txt files in the caption folder
    txt_files = glob.glob(os.path.join(caption_dir, "*.txt"))
    
    # Sort them numerically by the base filename
    # e.g., '1.txt' -> 1, '2.txt' -> 2, '10.txt' -> 10, etc.
    txt_files = sorted(
        txt_files,
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "image_filename"])

        for txt_path in txt_files:
            with open(txt_path, "r", encoding="utf-8") as txt_f:
                caption_text = txt_f.read().strip()

            base_name = os.path.splitext(os.path.basename(txt_path))[0]
            image_filename = base_name + ".jpg"

            writer.writerow([caption_text, image_filename])


if __name__ == "__main__":
    caption_folder = "D:/GitHub/multimodal/Urban1k/caption"
    image_folder   = "D:/GitHub/multimodal/Urban1k/image"
    output_csv     = "D:/GitHub/EffectiveTokenLength-MModels/urban1k.csv"

    preprocess_urban1k(caption_folder, image_folder, output_csv)
    print(f"CSV file created at: {output_csv}")
