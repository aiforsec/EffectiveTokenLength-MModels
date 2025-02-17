import json
import csv

def json_to_csv(input_json_path, output_csv_path):
    """Converts a JSON file to a CSV file with image paths and GPT responses."""
    with open(input_json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(["imagepath", "text"])
        
        for entry in data:
            image_path = entry.get("image", "")
            
            # Go through each conversation in this entry
            for conversation in entry.get("conversations", []):
                # Only pick messages where "from" is "gpt"
                if conversation.get("from") == "gpt":
                    text = conversation.get("value", "")
                    # Write a row: [image_path, gpt_text]
                    writer.writerow([image_path, text])

if __name__ == "__main__":
    input_json_path = "sharegpt4v_instruct_gpt4-vision_cap100k.json"  # Change this to your actual JSON file path
    output_csv_path = "sharegpt4v.csv"  # Change this to your desired CSV output path
    
    json_to_csv(input_json_path, output_csv_path)
    print(f"CSV file has been created successfully: {output_csv_path}")
