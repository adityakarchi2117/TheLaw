"""LLM-based legal text simplification with error handling."""

import os
import logging
import time
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SimplificationError(Exception):
    """Custom exception for simplification failures."""
    pass


@dataclass
class SimplificationConfig:
    """Configuration for the simplification module."""
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    max_tokens: int = 1500
    max_retries: int = 3
    retry_delay: float = 1.0


# Default configuration
DEFAULT_CONFIG = SimplificationConfig()


# Prompt templates

SIMPLIFICATION_PROMPT = """You are an expert legal translator who converts complex legal documents into clear, simple English that anyone can understand.

Your task: Simplify the following legal text while preserving ALL important information.

GUIDELINES:
1. Replace legal jargon with everyday words
2. Break long sentences into shorter ones
3. Explain any technical terms that must be kept
4. Maintain the same meaning and intent
5. Keep it concise but complete
6. Use bullet points for lists or multiple conditions
7. Highlight any important obligations, rights, or deadlines

LEGAL TEXT TO SIMPLIFY:
\"\"\"
{text}
\"\"\"

SIMPLIFIED VERSION (in plain English):"""


SHORT_TEXT_PROMPT = """Simplify this legal text into plain English. Be concise and clear:

"{text}"

Simple explanation:"""


# LLM client management

_groq_client = None


def _get_groq_client():
    """Get or create Groq client (lazy init)."""
    global _groq_client
    
    if _groq_client is not None:
        return _groq_client
    
    try:
        from groq import Groq
    except ImportError:
        raise SimplificationError(
            "Groq library not installed. Run: pip install groq"
        )
    
    # Try environment variable first
    api_key = os.getenv("GROQ_API_KEY")
    
    # Try .env file
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
        except ImportError:
            pass
    
    if not api_key:
        raise SimplificationError(
            "GROQ_API_KEY not found. Set it as environment variable or in .env file.\n"
            "Get your free API key at: https://console.groq.com/"
        )
    
    _groq_client = Groq(api_key=api_key)
    return _groq_client


def _call_llm_with_retry(
    prompt: str,
    config: SimplificationConfig = DEFAULT_CONFIG
) -> str:
    """Call LLM with retry logic for reliability."""
    client = _get_groq_client()
    last_error = None
    
    for attempt in range(config.max_retries):
        try:
            completion = client.chat.completions.create(
                model=config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that simplifies legal documents into plain English."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            response = completion.choices[0].message.content
            if response:
                return response.strip()
            else:
                raise SimplificationError("Empty response from LLM")
                
        except SimplificationError:
            raise
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (attempt + 1))
    
    raise SimplificationError(f"All LLM retry attempts failed. Last error: {last_error}")


# Public API

def build_prompt(text: str) -> str:
    """Build simplification prompt (adapts to text length)."""
    # Use shorter prompt for brief texts
    if len(text) < 500:
        return SHORT_TEXT_PROMPT.format(text=text)
    
    return SIMPLIFICATION_PROMPT.format(text=text)


def simplify_text(
    text: str,
    config: Optional[SimplificationConfig] = None
) -> Tuple[str, Optional[str]]:
    """Main entry: simplify legal text to plain English."""
    if not text or len(text.strip()) < 10:
        return "", "Text too short to simplify"
    
    config = config or DEFAULT_CONFIG
    
    try:
        prompt = build_prompt(text)
        simplified = _call_llm_with_retry(prompt, config)
        
        # Post-process: clean up any remaining formatting issues
        simplified = simplified.strip()
        
        # Remove any prompt leakage
        if simplified.lower().startswith("simplified version"):
            simplified = simplified.split(":", 1)[-1].strip()
        
        return simplified, None
        
    except SimplificationError as e:
        logger.error(f"Simplification failed: {e}")
        return "", str(e)
    except Exception as e:
        logger.exception("Unexpected error during simplification")
        return "", f"Unexpected error: {str(e)}"


def simplify_clause(clause: str) -> str:
    """Legacy: simplify clause (returns error msg on failure)."""
    simplified, error = simplify_text(clause)
    
    if error:
        # For legacy compatibility, return a fallback message
        logger.error(f"Clause simplification failed: {error}")
        return f"[Simplification unavailable: {error}]"
    
    return simplified


def check_api_connection() -> Tuple[bool, str]:
    """Test LLM API connection."""
    try:
        client = _get_groq_client()
        
        # Quick test call
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=5
        )
        
        if completion.choices[0].message.content:
            return True, "API connection successful"
        else:
            return False, "API returned empty response"
            
    except SimplificationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Connection failed: {str(e)}"
