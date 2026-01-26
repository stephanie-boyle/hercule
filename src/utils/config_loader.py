import json
import os

def find_path_dynamically(target_name):
    """
    Recursively search parent directories for a folder or file named target_name.
    Returns the full path if found, else None.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    while current_dir != os.path.dirname(current_dir):  # Stop at the root of the HD
        check_path = os.path.join(current_dir, target_name)
        if os.path.exists(check_path):
            return check_path
        current_dir = os.path.dirname(current_dir) # Move up one level
    
    return None

def load_disease_config(filename='disease_mapping.json'):
    """
    Load disease configuration from a JSON file.

    :param filename: Name of the JSON file to load.
    :return: List of disease configurations.
    """

    dict_dir = find_path_dynamically('dictionary')
    
    if not dict_dir:
        raise FileNotFoundError("Could not find the 'dictionary' folder in any parent directory.")

    file_path = os.path.join(dict_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{filename}' not found in {dict_dir}")

    with open(file_path, 'r') as f:
        diseases = json.load(f)

    return diseases