import os
import json
import csv

def extract_labels_from_json(json_folder):
    # Initialize a dictionary to store extracted labels by image ID
    extracted_labels = {}
    
    # Process each JSON file in the specified folder
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_file_path = os.path.join(json_folder, filename)
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                
                # Extract image ID and labels from JSON data
                image_id = data['image_id']
                dialog = data['dialog']
                
                # Extract labels from dialog messages
                labels = [msg['msg_t'] for msg in dialog if 'msg_t' in msg]
                labels_str = ', '.join(labels)  # Join labels into a single string
                
                # Store the extracted labels by image ID
                extracted_labels[str(image_id)] = labels_str
    
    return extracted_labels

def generate_labels_csv(input_folder, json_folder, output_csv_file):
    # Extract labels from JSON files
    extracted_labels = extract_labels_from_json(json_folder)
    
    # Create a list to store CSV rows
    csv_data = []
    
    # Process each image file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') and filename.startswith('Scene'):
            # Extract image ID from the filename (e.g., "Scene123.png" -> "123")
            image_id = filename.split('SceneScene')[1].split('.')[0]
            print(f"Extracted image ID: {image_id}, Filename: {filename}")
            
            if image_id in extracted_labels:
                labels = extracted_labels[image_id]
                
                # Append CSV row with image filename, image ID, and labels
                csv_data.append({'image_filename': filename, 'image_id': image_id, 'labels': labels})
            else:
                print(f"Labels not found for image ID: {image_id}")
    
    # Define CSV fieldnames based on the data structure
    fieldnames = ['image_filename', 'image_id', 'labels']
    
    # Write extracted data to CSV file
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

# Specify input folder, JSON folder, and output CSV file
input_folder = 'generated_scenes\generated_scenes'
json_folder = 'output\output'
output_csv_file = 'labels.csv'

# Generate CSV file based on image filenames and corresponding labels from JSON files
generate_labels_csv(input_folder, json_folder, output_csv_file)

print(f"CSV file generated: {output_csv_file}")
