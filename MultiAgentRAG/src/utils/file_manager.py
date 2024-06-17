# src/utils/file_manager.py

import os
import json
from typing import Dict
import logging

logger = logging.getLogger('file_manager')

def save_json(data: Dict, filename: str, output_dir: str = 'json_output'):
    """
    Save a dictionary as a JSON file.
    
    Args:
        data (Dict): The data to be saved as JSON.
        filename (str): The name of the JSON file.
        output_dir (str): The directory where the JSON file will be saved.
    
    Returns:
        None
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        logger.info(f"Saved JSON: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON {filename}: {e}")

def load_json(filename: str, output_dir: str = 'json_output') -> Dict:
    """
    Load a JSON file and return its content as a dictionary.
    
    Args:
        filename (str): The name of the JSON file.
        output_dir (str): The directory where the JSON file is located.
    
    Returns:
        Dict: The content of the JSON file.
    """
    try:
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'r') as json_file:
            data = json.load(json_file)
        
        logger.info(f"Loaded JSON: {output_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON {filename}: {e}")
        return {}
