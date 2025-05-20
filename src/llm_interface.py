import logging
from llama_cpp import Llama

logger = logging.getLogger(__name__)

def load_llm(model_path, model_config=None):
    """
    Loads a GGUF LLM model

    Args:
        model_path (str): Path to the .gguf model file.
        gguf_config (dict): Dictionary of parameters for loading the model.

    Returns:
        Llama: Loaded Llama object or None if an error occurs.
    """
    llm = None
    model_params = model_config or {}

    try:
        llm = Llama(
            model_path=model_path,
            **model_params,
        )
    except Exception as e:
        logger.error(f"Error loading GGUF model {model_path}: {e}")
        return None

    return llm

def generate_response(llm, messages, gen_config=None):
    """
    Generates a response from the loaded GGUF LLM using create_chat_completion.

    Args:
        llm (Llama): The loaded Llama object.
        messages (list): A list of message dictionaries, e.g.,
             [{"role": "user", "content": "Your prompt"}].
        gen_params (dict): Dictionary of generation parameters
            (e.g., max_tokens, temperature, top_p, stop).
                           
    Returns:
        str: The generated response text or None if an error occurs.
    """
    gen_params = gen_config or {}
    response_text = None

    try:
        response = llm.create_chat_completion(
            messages=messages,
            **gen_params,
        )
        response_text = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None
    
    return response_text
