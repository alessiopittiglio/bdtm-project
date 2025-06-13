import logging

logger = logging.getLogger(__name__)

def load_prompt(file_path: str) -> str | None:
    """
    Load the prompt template from a file.
    
    Args:
        file_path (str): The path to the prompt template file.
        
    Returns:
        str | None: The prompt template as a string, or None if the file does not exist.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template
    except FileNotFoundError:
        logger.error(f"Prompt template file '{file_path}' not found.")
        return None
