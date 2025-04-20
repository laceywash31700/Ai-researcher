from dotenv import load_dotenv
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 1. Force reload .env file
load_dotenv(override=True)

# 2. Explicitly verify the API key exists
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in .env file. "
        "Please add: OPENAI_API_KEY=your-actual-key"
    )

# 3. Initialize models with explicit auth
embedding_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY,  # Explicitly passed
    model="text-embedding-3-large",
    api_base="https://api.openai.com/v1",  # Explicit endpoint
    timeout=60,
    max_retries=3
)

llm_model = OpenAI(
    api_key=OPENAI_API_KEY,  # Explicitly passed
    model="gpt-4-turbo",  # Using valid model name
    temperature=0.3,
    timeout=60
)
