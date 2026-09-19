"""Centralized configuration management."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

# Path configuration

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# LLM configuration

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    
    # Groq (default provider - fast & free tier available)
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.3
    groq_max_tokens: int = 1500
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Token limits
    max_input_tokens: int = 4000
    chars_per_token: float = 4.0


# Detection configuration

@dataclass
class DetectionConfig:
    """Configuration for legal document detection."""
    
    # Minimum score to classify as legal
    min_legal_score: float = 5.0
    
    # Confidence thresholds
    confidence_definitely_legal: float = 0.75
    confidence_likely_legal: float = 0.50
    confidence_possibly_legal: float = 0.30


# Preprocessing configuration

@dataclass 
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    
    # Validation limits
    min_text_length: int = 50
    max_text_length: int = 100000
    
    # PDF settings
    max_pdf_pages: int = 100


# UI configuration

@dataclass
class UIConfig:
    """Configuration for Streamlit UI."""
    
    page_title: str = "Legal Document Simplifier"
    page_icon: str = "⚖️"
    layout: str = "wide"
    
    # Text area settings
    text_area_height: int = 250
    max_display_chars: int = 5000


# Global config instance

@dataclass
class Config:
    """Master configuration class."""
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Feature flags
    enable_clause_splitting: bool = False
    enable_risk_analysis: bool = False
    debug_mode: bool = False
    
    def __post_init__(self):
        """Load environment-based overrides."""
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config():
    """Reset configuration to defaults (useful for testing)."""
    global _config
    _config = None
