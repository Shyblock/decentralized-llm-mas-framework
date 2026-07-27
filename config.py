import os

# Which provider to use
PROVIDER = "groq"

# --------------------------------------------------
# Blockchain enforcement layer (Section IV / methodology Steps 2, 4, 10,
# 14-19). Set ENABLE_BLOCKCHAIN=false in .env to run purely off-chain
# (e.g. if Ganache isn't running) without touching this file.
# --------------------------------------------------
ENABLE_BLOCKCHAIN = os.getenv("ENABLE_BLOCKCHAIN", "true").lower() == "true"
GANACHE_URL = os.getenv("GANACHE_URL", "http://127.0.0.1:7545")

# Model for each provider
MODELS = {
#    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.5-flash",
    "github": "gpt-4.1-mini",
}

# NOTE: regardless of PROVIDER above, reflection.py's teacher model always
# runs on Groq (llama-3.3-70b-versatile) via llm_provider.generate_groq().
# This means GROQ_API_KEY_1 (or GROQ_API_KEY) must be set even if you set
# PROVIDER = "gemini" for the main agent generation loop.
