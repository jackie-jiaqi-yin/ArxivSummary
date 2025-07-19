import re
from datetime import datetime, timedelta, date
from pathlib import Path
import markdown2


def create_saved_title(title: str) -> str:
    """
    Create a safe filename from a paper title
    
    Args:
        title (str): The original paper title
        
    Returns:
        str: A cleaned title suitable for filenames
    """
    # replace space with "_"
    title = title.replace(' ', '_')
    # remove non-letter or non-digit characters
    cleaned_title = re.sub(r'[^a-zA-Z0-9_]', '', title)
    return cleaned_title


def format_date(date_str: str = None) -> date:
    """
    Format date string to date object
    
    Args:
        date_str (str, optional): Date string in 'YYYY-MM-DD' format
        
    Returns:
        date: A date object
    """
    if date_str is None:
        return datetime.now().date() - timedelta(days=3)
    else:
        return datetime.strptime(date_str, '%Y-%m-%d').date()


def clean_and_format_links(html_content: str) -> str:
    """
    Clean and format links in HTML content
    
    Args:
        html_content (str): HTML content with links
        
    Returns:
        str: Cleaned HTML content with properly formatted links
    """
    def format_link(match):
        url = match.group(1)
        # Remove any trailing characters that shouldn't be part of the URL
        url = re.sub(r'</?\w+>|&[a-z]+;|\s', '', url)
        url = re.sub(r'[),.\']$', '', url)
        return f'<a href="{url}" target="_blank">{url}</a>'

    # Pattern to match URLs, including those in Markdown links
    url_pattern = r'(?<!["=])(?:(?<=\()|\b)(https?://\S+?)(?=[),\s<]|$)'

    # First, handle any remaining plain text URLs
    html_content = re.sub(url_pattern, format_link, html_content)

    # Then, clean up any URLs that are already in HTML links
    html_content = re.sub(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', format_link, html_content)

    return html_content


def convert_md_to_html(md_file_path: str, html_file_path: str) -> str:
    """
    Convert markdown file to HTML with custom styling
    
    Args:
        md_file_path (str): Path to the markdown file
        html_file_path (str): Path to save the HTML file
        
    Returns:
        str: The generated HTML content
    """
    md_content = Path(md_file_path).read_text(encoding='utf-8')
    html_content = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'tables', 'break-on-newline'])
    html_content = clean_and_format_links(html_content)
    current_date = date.today().strftime("%B %d, %Y")

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Agent: Key ArXiv Papers Unveiled</title>
        <style>
            /* Your original styles here - unchanged */
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.4;
                color: #333;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
                font-size: 14px;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background-color: #ffffff;
            }}
            .header {{
                background-color: #2c3e50;
                color: #ffffff;
                padding: 15px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 22px;
                line-height: 1.2;
                color: #ffffff;
            }}
            .header p {{
                margin: 5px 0 0;
                font-size: 14px;
                font-style: italic;
                color: #ecf0f1;
            }}
            .content {{
                padding: 20px;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #2c3e50;
                margin-top: 0.8em;
                margin-bottom: 0.4em;
                line-height: 1.2;
            }}
            h1 {{ font-size: 1.6em; color: #2c3e50; }}
            h2 {{ font-size: 1.4em; color: #34495e; }}
            h3 {{ font-size: 1.3em; color: #34495e; }}
            h4 {{ font-size: 1.2em; }}
            h5 {{ font-size: 1.1em; }}
            h6 {{ font-size: 1em; }}
            p {{
                margin-top: 0.4em;
                margin-bottom: 0.4em;
            }}
            a {{
                color: #3498db;
                text-decoration: none;
                word-break: break-word;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .date {{
                text-align: center;
                font-style: italic;
                color: #7f8c8d;
                margin: 0;
                font-size: 0.9em;
                background-color: #ecf0f1;
                padding: 5px;
            }}
            pre {{
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                overflow-x: auto;
                font-size: 13px;
            }}
            code {{
                font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
                font-size: 13px;
                background-color: #f0f0f0;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 15px;
                font-size: 13px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 6px;
                text-align: left;
            }}
            th {{
                background-color: #34495e;
                color: #ffffff;
            }}
            ul, ol {{
                padding-left: 25px;
                margin-top: 0.4em;
                margin-bottom: 0.4em;
            }}
            li {{
                margin-bottom: 4px;
            }}
            .footer {{
                background-color: #34495e;
                padding: 15px;
                text-align: center;
                font-size: 14px;
                color: #ecf0f1;
                margin-top: 30px;
            }}
            .footer a {{
                color: #3498db;
                margin: 0 10px;
                font-weight: bold;
            }}
            .footer p {{
                margin: 10px 0;
            }}
            @media (max-width: 820px) {{
                .container {{
                    width: 100%;
                }}
                .content {{
                    padding: 15px;
                }}
                body {{
                    font-size: 14px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <table border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #2c3e50; color: #ffffff; text-align: center;">
                <tr>
                    <td style="padding: 15px 15px 10px 15px;">
                        <h1 style="margin: 0; font-size: 22px; line-height: 1.2; color: #ffffff;">AI Evolution: Key ArXiv Papers Unveiled</h1>
                        <p style="margin: 5px 0 0; font-size: 14px; font-style: italic; color: #ecf0f1;">Cutting-edge research at your fingertips</p>
                    </td>
                </tr>
            </table>
            <p class="date" style="text-align: center; font-style: italic; color: #7f8c8d; margin: 0; font-size: 0.9em; background-color: #ecf0f1; padding: 5px;">Generated on {current_date}</p>
            <div class="content">
                {html_content}
            </div>
            <table border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #34495e; color: #ecf0f1; text-align: center; font-size: 14px; margin-top: 30px;">
                <tr>
                    <td style="padding: 15px;">
                        <div style="display: flex; justify-content: center;">
                            <div style="display: inline-block; white-space: nowrap;">
                                <a href="https://idwebelements.microsoft.com/GroupManagement.aspx?Group=amplify-ai-digest&Operation=join" style="color: #3498db; text-decoration: none; font-weight: bold; padding: 0 5px;">Join email list</a>
                                <span style="color: #ecf0f1; padding: 0 5px;">|</span>
                                <a href="https://idwebelements.microsoft.com/GroupManagement.aspx?Group=amplify-ai-digest&Operation=leave" style="color: #3498db; text-decoration: none; font-weight: bold; padding: 0 5px;">Unsubscribe</a>
                            </div>
                        </div>
                        <p style="margin: 15px 0 0 0;">Users must be connected to the corporate network (such as MSFTVPN-Manual) in order for the links to work.</p>
                    </td>
                </tr>
            </table>
        </div>
    </body>
    </html>
    """

    Path(html_file_path).write_text(html_template, encoding='utf-8')
    print(f"HTML file created: {html_file_path}")
    return html_template