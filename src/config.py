from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths configuration
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.resolve()

DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
MODELS_DIR = BASE_DIR / "models"
PROMPTS_DIR = BASE_DIR / "prompts"

MCQA_GENERATED_JSON = DATA_DIR / "mcqa_generated.json"
MCQA_CURATED_JSON = DATA_DIR / "mcqa_curated.json"

# --------------------------------------------------------------------------------------
# Embedding model configuration
# --------------------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "university_lectures"

# --------------------------------------------------------------------------------------
# Text chunking configuration
# --------------------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --------------------------------------------------------------------------------------
# MCQA generation configuration
# --------------------------------------------------------------------------------------
LLM_MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M"
LLM_MODEL_PATH = MODELS_DIR / "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
PROMPT_TEMPLATE_PATH = PROMPTS_DIR / "mcqa_generation_template.txt"

NUM_CHUNKS_TO_SAMPLE = 3 # per lecture

LLM_MODEL_CONFIG = {
    'n_gpu_layers': 61,
    'temp': 0.6,
    'top_p': 0.95,
    'top_k': 20,
    'min_p': 0,
    'n_ctx': 16384,
    'flash_attn': True,
    'verbose': False
}

LLM_GENERATION_CONFIG = {
    'max_tokens': None,
    'temperature': 0.6,
}

# --------------------------------------------------------------------------------------
# Placeholder for MCQA evaluation config
# --------------------------------------------------------------------------------------

