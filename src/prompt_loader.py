import os
import logging
from src import config

logger = logging.getLogger(__name__)

def load_prompt(template_filename: str) -> str | None:
    """
    Load the prompt template from a file.
    
    Args:
        template_filename (str): The path to the prompt template file.
        
    Returns:
        str | None: The prompt template as a string, or None if the file does not exist.
    """
    file_path = os.path.join(config.PROMPTS_DIR, template_filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template
    except FileNotFoundError:
        logger.error(f"Prompt template file '{template_filename}' not found.")
        return None
