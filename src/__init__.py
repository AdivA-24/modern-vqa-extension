"""
Modern VQA Extension
Cross-Lingual Visual Question Answering with Modern Vision-Language Models

Comparing ViLT (2021) to LLaVA-1.6, Qwen-VL, and BLIP-2 (2024-2025)
"""

__version__ = "1.0.0"
__author__ = "Adiv Ahsan"

from .models import ModernVQA, LegacyVQA, BLIP2VQA, get_model
from .pipeline import CrossLingualVQAPipeline, create_pipeline
from .translation import (
    translate_to_english,
    translate_from_english,
    detect_language,
    get_supported_languages,
    TranslationError
)

__all__ = [
    # Models
    "ModernVQA",
    "LegacyVQA",
    "BLIP2VQA",
    "get_model",
    # Pipeline
    "CrossLingualVQAPipeline",
    "create_pipeline",
    # Translation
    "translate_to_english",
    "translate_from_english",
    "detect_language",
    "get_supported_languages",
    "TranslationError",
]
