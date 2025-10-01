"""
Translation utilities for Cross-Lingual VQA
Uses Google Translate API for multilingual support (100+ languages)
"""

from typing import Tuple


class TranslationError(Exception):
    """Custom exception for translation errors"""
    pass


def translate_to_english(text: str) -> Tuple[str, str]:
    """
    Translate text from any language to English with language detection

    Args:
        text: Input text in any language

    Returns:
        Tuple of (translated_text, source_language_code)

    Examples:
        >>> translate_to_english("¿Qué están haciendo los gatos?")
        ('What are the cats doing?', 'es')

        >>> translate_to_english("What are they doing?")
        ('What are they doing?', 'en')
    """
    try:
        from googletrans import Translator

        translator = Translator()
        result = translator.translate(text, dest='en')

        return result.text, result.src

    except ImportError:
        raise TranslationError(
            "googletrans library not installed. Install with: pip install googletrans==4.0.0rc1"
        )
    except Exception as e:
        raise TranslationError(f"Translation to English failed: {str(e)}")


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate text from English to target language

    Args:
        text: Input text in English
        target_lang: Target language code (e.g., 'fr', 'es', 'de', 'zh-cn')

    Returns:
        Translated text in target language

    Examples:
        >>> translate_from_english("The cats are sleeping.", "fr")
        'Les chats dorment.'

        >>> translate_from_english("The cats are sleeping.", "es")
        'Los gatos están durmiendo.'
    """
    if target_lang == 'en':
        return text

    try:
        from googletrans import Translator

        translator = Translator()
        result = translator.translate(text, src='en', dest=target_lang)

        return result.text

    except ImportError:
        raise TranslationError(
            "googletrans library not installed. Install with: pip install googletrans==4.0.0rc1"
        )
    except Exception as e:
        raise TranslationError(f"Translation from English to {target_lang} failed: {str(e)}")


def detect_language(text: str) -> str:
    """
    Detect the language of input text

    Args:
        text: Input text

    Returns:
        Language code (e.g., 'en', 'fr', 'es')

    Examples:
        >>> detect_language("Hello, how are you?")
        'en'

        >>> detect_language("Bonjour, comment allez-vous?")
        'fr'
    """
    try:
        from googletrans import Translator

        translator = Translator()
        result = translator.detect(text)

        return result.lang

    except ImportError:
        raise TranslationError(
            "googletrans library not installed. Install with: pip install googletrans==4.0.0rc1"
        )
    except Exception as e:
        raise TranslationError(f"Language detection failed: {str(e)}")


# Supported language codes (subset of most common languages)
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'pl': 'Polish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'fi': 'Finnish',
    'no': 'Norwegian',
}


def get_supported_languages():
    """Get dictionary of supported language codes and names"""
    return SUPPORTED_LANGUAGES.copy()
