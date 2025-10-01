"""
Cross-Lingual VQA Pipeline
Orchestrates translation, VQA inference, and answer formatting
"""

from PIL import Image
from typing import Dict, Optional, Union
import os

from .models import ModernVQA, LegacyVQA, BLIP2VQA, get_model
from .translation import translate_to_english, translate_from_english, TranslationError


class CrossLingualVQAPipeline:
    """
    Complete pipeline for cross-lingual Visual Question Answering

    Supports:
    - Multilingual question input (100+ languages)
    - Multiple VQA models (Modern/Legacy/BLIP2)
    - Cross-lingual answer generation
    - Optional reasoning explanations
    """

    def __init__(self, model_type: str = "modern", enable_reasoning: bool = True):
        """
        Initialize the cross-lingual VQA pipeline

        Args:
            model_type: Type of VQA model to use ("modern", "legacy", or "blip2")
            enable_reasoning: Whether to include reasoning in answers (modern models only)
        """
        print(f"Initializing Cross-Lingual VQA Pipeline with {model_type} model...")

        self.model_type = model_type
        self.enable_reasoning = enable_reasoning and (model_type == "modern")

        # Load VQA model
        self.vqa_model = get_model(model_type)

        print(f"✓ Pipeline ready with {model_type} model")

    def query(
        self,
        image: Union[Image.Image, str],
        question: str,
        target_lang: Optional[str] = None,
        explain: Optional[bool] = None
    ) -> Dict[str, str]:
        """
        Process a cross-lingual VQA query

        Args:
            image: PIL Image object or path to image file
            question: Question in any language
            target_lang: Target language for answer (if None, uses source language)
            explain: Whether to include reasoning (overrides instance setting)

        Returns:
            Dictionary with:
                - question_original: Original question
                - question_english: Translated English question
                - answer_english: Answer in English (if available)
                - answer: Answer in target language
                - source_lang: Detected source language code
                - target_lang: Output language code
                - model_used: VQA model type

        Examples:
            >>> pipeline = CrossLingualVQAPipeline("modern")
            >>> result = pipeline.query(image, "¿Qué están haciendo los gatos?")
            >>> print(result['answer'])  # Spanish answer
            "Los gatos están durmiendo en el sofá."
        """
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image).convert('RGB')

        # Determine whether to use reasoning
        use_reasoning = explain if explain is not None else self.enable_reasoning

        try:
            # Step 1: Translate question to English
            print(f"Original question: {question}")
            en_question, source_lang = translate_to_english(question)
            print(f"English translation: {en_question} (detected: {source_lang})")

            # Step 2: Get VQA answer
            print(f"Running {self.model_type} VQA model...")
            if self.model_type == "modern" and use_reasoning:
                answer_en = self.vqa_model.answer(image, en_question, explain=True)
            else:
                answer_en = self.vqa_model.answer(image, en_question)

            print(f"English answer: {answer_en}")

            # Step 3: Translate answer to target language
            output_lang = target_lang or source_lang

            if output_lang == 'en':
                answer_translated = answer_en
            else:
                print(f"Translating to {output_lang}...")
                answer_translated = translate_from_english(answer_en, output_lang)
                print(f"Translated answer: {answer_translated}")

            return {
                'question_original': question,
                'question_english': en_question,
                'answer_english': answer_en if output_lang != 'en' else None,
                'answer': answer_translated,
                'source_lang': source_lang,
                'target_lang': output_lang,
                'model_used': self.model_type
            }

        except TranslationError as e:
            return {
                'question_original': question,
                'error': f"Translation error: {str(e)}",
                'model_used': self.model_type
            }
        except Exception as e:
            return {
                'question_original': question,
                'error': f"VQA error: {str(e)}",
                'model_used': self.model_type
            }

    def compare_models(
        self,
        image: Union[Image.Image, str],
        question: str,
        target_lang: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Compare answers from multiple VQA models

        Args:
            image: PIL Image object or path to image file
            question: Question in any language
            target_lang: Target language for answer

        Returns:
            Dictionary with results from each model:
                {
                    'modern': {...},
                    'legacy': {...},
                    'blip2': {...}
                }
        """
        results = {}

        for model_type in ['modern', 'legacy', 'blip2']:
            try:
                print(f"\n--- Testing {model_type} model ---")
                pipeline = CrossLingualVQAPipeline(model_type)
                result = pipeline.query(image, question, target_lang)
                results[model_type] = result
            except Exception as e:
                print(f"Error with {model_type} model: {e}")
                results[model_type] = {
                    'error': str(e),
                    'model_used': model_type
                }

        return results


def create_pipeline(model_type: str = "modern", **kwargs) -> CrossLingualVQAPipeline:
    """
    Factory function to create a VQA pipeline

    Args:
        model_type: Type of model ("modern", "legacy", or "blip2")
        **kwargs: Additional arguments for CrossLingualVQAPipeline

    Returns:
        Initialized pipeline
    """
    return CrossLingualVQAPipeline(model_type=model_type, **kwargs)
