import json 
import pandas as pd
import ast


def read_tsv_column(file_path, column_name, dtype='text'):
    try:
        # Read the TSV file into a pandas DataFrame
        df = pd.read_csv(file_path, sep='\t')

        # Check if the specified column exists
        if column_name in df.columns:
            # Extract the specified column
            column_data = list(df[column_name])
            if dtype == "list":
                column_data = [ast.literal_eval(item) for item in column_data]
            return column_data
        else:
            return f"Column '{column_name}' not found in the TSV file."
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Error reading TSV file: {e}"

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_to_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)