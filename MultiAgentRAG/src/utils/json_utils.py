# src/utils/json_utils.py

import os
import json
from typing import Dict
import logging

logger = logging.getLogger('json_utils')

def save_memory_to_json(memory_data: Dict[str, str], output_dir: str = 'json_output'):
    """
    Saves the structured memory data to a JSON file.
    
    Args:
        memory_data (Dict[str, str]): Dictionary containing the structured memory data.
        output_dir (str): Directory to save the JSON files.
    
    Returns:
        None
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, f"{memory_data['title']}.json")
        
        with open(output_path, 'w') as json_file:
            json.dump(memory_data, json_file, indent=4)
        
        logger.info(f"Generated JSON: {output_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file {memory_data['title']}: {e}")
