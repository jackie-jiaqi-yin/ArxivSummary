import os
from src.utils.helpers import convert_md_to_html


class HTMLGenerator:
    def __init__(self):
        pass
    
    def generate_html(self, markdown_file: str, output_dir: str) -> str:
        """
        Generate HTML from markdown summary
        
        Args:
            markdown_file (str): Path to markdown file
            output_dir (str): Directory to save HTML file
            
        Returns:
            str: Path to generated HTML file
        """
        if not os.path.exists(markdown_file):
            raise FileNotFoundError(f"Markdown file not found: {markdown_file}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        html_file = os.path.join(output_dir, 'abstract_summary.html')
        html_content = convert_md_to_html(markdown_file, html_file)
        
        return html_file
    
    def run(self, markdown_file: str, output_dir: str) -> str:
        """
        Run the HTML generation process
        
        Args:
            markdown_file (str): Path to markdown file
            output_dir (str): Directory to save HTML file
            
        Returns:
            str: Path to generated HTML file
        """
        print('Generating HTML...')
        return self.generate_html(markdown_file, output_dir)