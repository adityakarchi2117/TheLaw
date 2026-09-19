"""Centralised configuration — single source of truth for all settings.

Loads environment variables from .env (local) or Streamlit Secrets (cloud).
"""

import os
import logging
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


def _get_env(key: str, default: str = "") -> str:
    """Read from os.environ first, then Streamlit secrets (cloud deployment fallback)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
MODELS_DIR      = PROJECT_ROOT / "models"
DATA_DIR        = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# ─────────────────────────────────────────────
# LLM Configuration (Groq & Google Gemini)
# ─────────────────────────────────────────────
GROQ_API_KEY:    str   = _get_env("GROQ_API_KEY")
GEMINI_API_KEY:  str   = _get_env("GEMINI_API_KEY") or _get_env("GOOGLE_API_KEY")

LLM_PROVIDER:    str   = _get_env("LLM_PROVIDER", "auto").lower()

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
}

AVAILABLE_MODELS = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "openai/gpt-oss-20b",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ],
    "gemini": [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
}

raw_model = _get_env("LLM_MODEL", "")
if raw_model in ("llama-3.1-8b-instant", "llama3-8b-instant"):
    logger.warning("llama-3.1-8b-instant is deprecated by Groq. Upgrading default to llama-3.3-70b-versatile.")
    LLM_MODEL: str = "llama-3.3-70b-versatile"
else:
    LLM_MODEL: str = raw_model or DEFAULT_MODELS["groq"]

LLM_TEMPERATURE: float = float(_get_env("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS:  int   = int(_get_env("LLM_MAX_TOKENS", "2048"))

# ─────────────────────────────────────────────
# AWS / S3
# ─────────────────────────────────────────────
AWS_ACCESS_KEY_ID:     str = _get_env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY: str = _get_env("AWS_SECRET_ACCESS_KEY")
AWS_REGION:            str = _get_env("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME:        str = _get_env("S3_BUCKET_NAME")

# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────
EMBEDDING_MODEL:  str = _get_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE: str = _get_env("EMBEDDING_DEVICE", "cpu")

# ─────────────────────────────────────────────
# RAG / Retrieval
# ─────────────────────────────────────────────
CHUNK_SIZE:      int = int(_get_env("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP:   int = int(_get_env("CHUNK_OVERLAP", "200"))
RETRIEVER_TOP_K: int = int(_get_env("RETRIEVER_TOP_K", "4"))

# ─────────────────────────────────────────────
# Legal Detection
# ─────────────────────────────────────────────
MIN_LEGAL_SCORE:            float = float(_get_env("MIN_LEGAL_SCORE", "5.0"))
CONFIDENCE_THRESHOLD_LEGAL: float = float(_get_env("CONFIDENCE_THRESHOLD_LEGAL", "0.50"))


def normalize_model_name(provider: str, model_name: str) -> str:
    """Normalize and upgrade deprecated model names."""
    if not model_name:
        return DEFAULT_MODELS.get(provider, "llama-3.3-70b-versatile")
    if provider == "groq" and model_name in ("llama-3.1-8b-instant", "llama3-8b-instant"):
        logger.warning("llama-3.1-8b-instant is deprecated by Groq. Upgrading to llama-3.3-70b-versatile.")
        return "llama-3.3-70b-versatile"
    return model_name


def _get_api_key_for_provider(provider: str) -> str:
    """Retrieve the API key for a given provider, checking session state then environment."""
    try:
        import streamlit as st
        ui_key = st.session_state.get(f"{provider}_api_key_input", "")
        if ui_key and str(ui_key).strip():
            return str(ui_key).strip()
    except Exception:
        pass

    if provider == "gemini":
        return (_get_env("GEMINI_API_KEY") or _get_env("GOOGLE_API_KEY")).strip()
    elif provider == "groq":
        return _get_env("GROQ_API_KEY").strip()
    return ""


def get_active_provider_and_model(provider: str = None, model: str = None) -> tuple[str, str]:
    """Resolve the active LLM provider and model name."""
    session_prov = ""
    session_model = ""
    try:
        import streamlit as st
        session_prov = st.session_state.get("selected_llm_provider", "")
        session_model = st.session_state.get("selected_llm_model", "")
    except Exception:
        pass

    prov = (provider or session_prov or LLM_PROVIDER or "auto").lower().strip()

    if prov == "auto":
        gemini_key = _get_api_key_for_provider("gemini")
        groq_key = _get_api_key_for_provider("groq")
        if gemini_key and not groq_key:
            prov = "gemini"
        else:
            prov = "groq"

    if prov not in ("groq", "gemini"):
        prov = "groq"

    mod = model or session_model or LLM_MODEL or DEFAULT_MODELS.get(prov, "llama-3.3-70b-versatile")
    mod = normalize_model_name(prov, mod)

    # Ensure model matches provider
    if prov == "gemini" and mod not in AVAILABLE_MODELS.get("gemini", []):
        mod = DEFAULT_MODELS["gemini"]
    elif prov == "groq" and mod not in AVAILABLE_MODELS.get("groq", []):
        mod = DEFAULT_MODELS["groq"]

    return prov, mod


def _validate_api_key(provider: str) -> str:
    """Validate and return the API key for the chosen provider."""
    key = _get_api_key_for_provider(provider)
    if not key:
        if provider == "gemini":
            raise EnvironmentError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) not found.\n"
                "• Add it to .env (local) or Streamlit Secrets (cloud)\n"
                "• Or enter it in the sidebar settings\n"
                "• Get a free key at: https://aistudio.google.com/app/apikey"
            )
        else:
            raise EnvironmentError(
                "GROQ_API_KEY not found.\n"
                "• Add it to .env (local) or Streamlit Secrets (cloud)\n"
                "• Or enter it in the sidebar settings\n"
                "• Get a free key at: https://console.groq.com/keys"
            )
    return key


_LLM_CACHE = {}


def get_llm(provider: str = None, model: str = None, api_key: str = None, temperature: float = None):
    """Return a cached LLM instance configured for the active provider (Groq or Google Gemini)."""
    prov, mod = get_active_provider_and_model(provider, model)
    key = api_key or _validate_api_key(prov)
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    cache_key = (prov, mod, hash(key), temp)
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    logger.info(f"Initializing LLM: provider={prov}, model={mod}, temp={temp}")

    if prov == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        instance = ChatGoogleGenerativeAI(
            model=mod,
            google_api_key=key,
            temperature=temp,
            max_output_tokens=LLM_MAX_TOKENS,
        )
    elif prov == "groq":
        from langchain_groq import ChatGroq
        instance = ChatGroq(
            api_key=key,
            model_name=mod,
            temperature=temp,
            max_tokens=LLM_MAX_TOKENS,
            max_retries=3,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {prov}")

    _LLM_CACHE[cache_key] = instance
    return instance


def clear_llm_cache():
    """Clear cached LLM instances."""
    _LLM_CACHE.clear()


def check_api_connection(provider: str = None, model: str = None) -> tuple[bool, str]:
    """Quick health check for the active or requested LLM API."""
    try:
        prov, mod = get_active_provider_and_model(provider, model)
        llm = get_llm(provider=prov, model=mod)
        resp = llm.invoke("Say OK")
        if resp and resp.content:
            return True, f"Connected to {prov.title()} ({mod})"
        return False, f"{prov.title()} returned empty response"
    except Exception as e:
        prov, mod = get_active_provider_and_model(provider, model)
        return False, f"{prov.title()} ({mod}) failed: {e}"


def get_aws_status() -> dict:
    """Return a dict with AWS/S3 configuration status."""
    return {
        "configured": all([
            _get_env("AWS_ACCESS_KEY_ID"),
            _get_env("AWS_SECRET_ACCESS_KEY"),
            _get_env("S3_BUCKET_NAME"),
        ]),
        "bucket": _get_env("S3_BUCKET_NAME"),
        "region": _get_env("AWS_REGION", "ap-south-1"),
    }
