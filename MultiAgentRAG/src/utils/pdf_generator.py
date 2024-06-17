# src/utils/pdf_generator.py

from fpdf import FPDF
from typing import Dict
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Memory Document', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(memory_data: Dict[str, str], output_dir: str = 'output'):
    """
    Generates a PDF document from the provided memory data.
    
    Args:
        memory_data (Dict[str, str]): Dictionary containing the structured memory data.
        output_dir (str): Directory to save the generated PDF documents.
    
    Returns:
        None
    """
    pdf = PDF()
    pdf.add_page()
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=memory_data['title'], ln=True, align='C')

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, txt=f"Query: {memory_data['query']}")
    pdf.multi_cell(0, 10, txt=f"Result: {memory_data['result']}")
    pdf.multi_cell(0, 10, txt=f"Tags: {memory_data['tags']}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{memory_data['title']}.pdf")
    pdf.output(output_path)
    print(f"Generated PDF: {output_path}")
