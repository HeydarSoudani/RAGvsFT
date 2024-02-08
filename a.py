
import json
import os

# Directory containing your JSON files
directory_path = 'component0_preprocessing/generated_data/popQA_EQformat/entity'

# Directory where you want to save the organized JSON files
output_directory = 'component0_preprocessing/generated_data/popQA_EQformat/entity2'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a JSON file
    if filename.endswith('.json'):
        # Construct the full path to the file
        file_path = os.path.join(directory_path, filename)
        
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            # Parse the JSON-like string into a Python object
            data = json.load(file)
        
        # Construct the path for the organized JSON file
        output_file_path = os.path.join(output_directory, filename)
        
        # Write the Python object back to a new JSON file with indentation
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
            
        print(f"Organized {filename} and saved to {output_file_path}")

print("All files have been processed.")


