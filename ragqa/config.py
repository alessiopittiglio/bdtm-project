from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths configuration
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.resolve()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
QA_DIR = DATA_DIR / "qa"
NIAH_DIR = DATA_DIR / "niah"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
MODELS_DIR = BASE_DIR / "models"
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "outputs"

MCQA_GENERATED_JSON = QA_DIR / "mcqa_generated.json"
MCQA_CURATED_JSON = QA_DIR / "mcqa_curated.json"

# --------------------------------------------------------------------------------------
# Cleaning configuration
# --------------------------------------------------------------------------------------
MIN_SHORT_TOKEN_REPETITIONS = 5
MAX_SHORT_TOKEN_LENGTH = 2

# --------------------------------------------------------------------------------------
# Embedding model configuration
# --------------------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "university_lectures"

# --------------------------------------------------------------------------------------
# Text chunking configuration
# --------------------------------------------------------------------------------------
GENERATION_CHUNK_SIZE = 1000
GENERATION_CHUNK_OVERLAP = 200

RETRIEVAL_CHUNK_SIZE = 384
RETRIEVAL_CHUNK_OVERLAP = 64

# --------------------------------------------------------------------------------------
# MCQA generation configuration
# --------------------------------------------------------------------------------------
GENERATION_MODEL_PATH = MODELS_DIR / "gpt-oss-20b-mxfp4.gguf"
GENERATION_PROMPT_PATH = PROMPTS_DIR / "mcqa_generation.txt"
SELECTION_PROMPT_PATH = PROMPTS_DIR / "chunk_selection.txt"

LLM_MODEL_CONFIG = {
    "n_gpu_layers": 23,
    "temp": 1.0,
    "top_p": 1,
    "top_k": 0,
    "min_p": 0,
    "n_ctx": 32768,  # max 128000
    "flash_attn": True,
    "swa_full": False,
    "verbose": False,
}

LLM_GENERATION_CONFIG = {
    "max_tokens": None,
    "temperature": 1.0,
}

# --------------------------------------------------------------------------------------
# MCQA evaluation config
# --------------------------------------------------------------------------------------
ANSWER_FORMAT_PROMPT = PROMPTS_DIR / "mcqa_answer.txt"
