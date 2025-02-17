import json
import csv
import os

def json_to_csv(input_json_path, output_csv_path):
    """Converts a JSON file to a CSV file with image paths and GPT responses."""
    with open(input_json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(["image_filename", "document"])
        
        for entry in data:
            image_path = entry.get("image", "")
            
            # Go through each conversation in this entry
            for conversation in entry.get("conversations", []):
                # Only pick messages where "from" is "gpt"
                if conversation.get("from") == "gpt":
                    text = conversation.get("value", "")
                    # Write a row: [image_path, gpt_text]
                    writer.writerow([image_path, text])

def remove_missing_images(csv_path, image_dir):
    """Removes rows from the CSV where the image file is missing in the given image directory."""
    temp_csv_path = csv_path + "_temp"
    
    with open(csv_path, 'r', encoding='utf-8') as infile, open(temp_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read and write the header
        header = next(reader)
        writer.writerow(header)
        
        for row in reader:
            image_relative_path = row[0]  # e.g., "wikiart/images/adriaen-brouwer_inn-with-drunken-peasants.jpg"
            image_abs_path = os.path.join(image_dir, os.path.basename(image_relative_path))  # Get full path

            if os.path.exists(image_abs_path):  # Check if the image file exists in the directory
                writer.writerow(row)

    # Replace the original file with the cleaned one
    os.replace(temp_csv_path, csv_path)
    print(f"Rows with missing images removed. Updated CSV saved as: {csv_path}")

if __name__ == "__main__":
    input_json_path = "input.json"  # Change this to your actual JSON file path
    output_csv_path = "output.csv"  # Change this to your desired CSV output path
    image_dir = "images/"  # Change this to the directory where your images are stored
    
    json_to_csv(input_json_path, output_csv_path)
    print(f"CSV file has been created: {output_csv_path}")

    remove_missing_images(output_csv_path, image_dir)
    print("CSV file has been cleaned by removing rows with missing images.")