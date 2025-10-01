"""
VQA Model Wrappers for Cross-Lingual VQA
Supports both legacy (ViLT 2021) and modern (LLaVA-1.6 2024) models
"""

import torch
from PIL import Image
from typing import Optional


class ModernVQA:
    """
    Wrapper for modern Vision-Language Models (2024-2025)
    Uses LLaVA-1.6 for improved reasoning and multilingual support
    """

    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        """
        Initialize modern VLM
        Args:
            model_name: HuggingFace model identifier
        """
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

            print(f"Loading modern VLM: {model_name}")
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.model_name = model_name
            print(f"✓ Successfully loaded {model_name}")

        except Exception as e:
            print(f"Warning: Could not load {model_name}: {e}")
            print("Falling back to CPU inference...")
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            self.model_name = model_name

    def answer(self, image: Image.Image, question: str, explain: bool = False) -> str:
        """
        Get answer from modern VLM with optional reasoning

        Args:
            image: PIL Image object
            question: Question in English
            explain: If True, request reasoning explanation

        Returns:
            answer: String response from the model
        """
        if explain:
            prompt = f"USER: <image>\n{question} Explain your reasoning briefly.\nASSISTANT:"
        else:
            prompt = f"USER: <image>\n{question}\nASSISTANT:"

        try:
            inputs = self.processor(prompt, image, return_tensors="pt")

            # Move to device if using GPU
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

            answer = self.processor.decode(output[0], skip_special_tokens=True)

            # Extract only the assistant's response
            if "ASSISTANT:" in answer:
                answer = answer.split("ASSISTANT:")[-1].strip()

            return answer

        except Exception as e:
            return f"Error generating answer: {str(e)}"


class LegacyVQA:
    """
    Wrapper for legacy ViLT model (2021)
    For comparison with modern models
    """

    def __init__(self, model_name: str = "dandelin/vilt-b32-finetuned-vqa"):
        """
        Initialize ViLT model
        Args:
            model_name: HuggingFace model identifier
        """
        from transformers import ViltProcessor, ViltForQuestionAnswering

        print(f"Loading legacy VQA model: {model_name}")
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.model = ViltForQuestionAnswering.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.model_name = model_name
        print(f"✓ Successfully loaded {model_name}")

    def answer(self, image: Image.Image, question: str) -> str:
        """
        Get answer from ViLT model

        Args:
            image: PIL Image object
            question: Question in English

        Returns:
            answer: Short answer string (single word/phrase)
        """
        try:
            inputs = self.processor(image, question, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = self.model.config.id2label[idx]

            return answer

        except Exception as e:
            return f"Error: {str(e)}"


class BLIP2VQA:
    """
    Wrapper for BLIP-2 model (2023)
    Additional baseline for comparison
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        """
        Initialize BLIP-2 model
        Args:
            model_name: HuggingFace model identifier
        """
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        print(f"Loading BLIP-2 model: {model_name}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model_name = model_name
        print(f"✓ Successfully loaded {model_name}")

    def answer(self, image: Image.Image, question: str) -> str:
        """
        Get answer from BLIP-2 model

        Args:
            image: PIL Image object
            question: Question in English

        Returns:
            answer: Generated answer string
        """
        try:
            inputs = self.processor(image, question, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return answer.strip()

        except Exception as e:
            return f"Error: {str(e)}"


def get_model(model_type: str = "modern"):
    """
    Factory function to get VQA model by type

    Args:
        model_type: One of "modern", "legacy", "blip2"

    Returns:
        VQA model instance
    """
    if model_type == "modern":
        return ModernVQA()
    elif model_type == "legacy":
        return LegacyVQA()
    elif model_type == "blip2":
        return BLIP2VQA()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: modern, legacy, blip2")
