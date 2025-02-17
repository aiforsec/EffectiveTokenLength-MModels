import pandas as pd

def preprocess_roco(input_csv, output_csv):
    # Read the original CSV
    df = pd.read_csv(input_csv, encoding="utf-8")
    
    # Rename columns: caption -> document, name -> image_filename
    df.rename(columns={"caption": "document", "name": "image_filename"}, inplace=True)
    
    # Drop the 3rd column (index 2)
    # (If you know the exact column name, you can use df.drop(columns=["some_column"]) instead.)
    df.drop(columns=["id"])
    
    # Write the updated DataFrame to a new CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")

if __name__ == "__main__":
    input_csv_path = "path/to/your/input.csv"  # Replace with your actual input CSV file path
    output_csv_path = "path/to/your/roco.csv"  # Replace with your desired output CSV file path

    preprocess_roco(input_csv_path, output_csv_path)
    print(f"Preprocessed CSV saved to: {output_csv_path}")
