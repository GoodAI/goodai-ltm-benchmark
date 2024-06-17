# src/utils/pdf_to_json.py

import os
import PyPDF2
import json

def pdf_to_json(pdf_path: str, output_dir: str = 'json_output'):
    """
    Converts a PDF file to a JSON file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Directory to save the JSON files.
    
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfFileReader(pdf_file)
        text = ""
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
    
    # Assume the PDF filename follows the format 'Memory_X.pdf'
    base_name = os.path.basename(pdf_path)
    title = os.path.splitext(base_name)[0]
    memory_data = {
        "title": title,
        "content": text,
        "tags": "converted"
    }
    
    output_path = os.path.join(output_dir, f"{title}.json")
    
    with open(output_path, 'w') as json_file:
        json.dump(memory_data, json_file, indent=4)
    
    print(f"Converted PDF to JSON: {output_path}")