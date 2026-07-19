"""
Gradio Demo for Modern Cross-Lingual VQA
Compare ViLT (2021) vs Modern VLMs (LLaVA-1.6, 2024)
"""

import gradio as gr
import sys
import os

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.pipeline import CrossLingualVQAPipeline
except ImportError:
    print("Warning: Could not import pipeline. Make sure src/ is in the path.")


# Initialize pipelines
print("Loading VQA models...")
try:
    modern_pipeline = CrossLingualVQAPipeline("modern", enable_reasoning=True)
    legacy_pipeline = CrossLingualVQAPipeline("legacy", enable_reasoning=False)
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False


def compare_vqa(image, question, target_lang, include_reasoning):
    """
    Compare VQA results from modern and legacy models

    Args:
        image: PIL Image
        question: Question string
        target_lang: Target language code
        include_reasoning: Whether to include explanations

    Returns:
        Formatted comparison string
    """
    if not models_loaded:
        return "⚠️ Models not loaded. Please check the console for errors."

    if image is None:
        return "⚠️ Please upload an image first."

    if not question or len(question.strip()) == 0:
        return "⚠️ Please enter a question."

    try:
        # Get results from both models
        print(f"\n=== Modern Model ===")
        modern_result = modern_pipeline.query(
            image,
            question,
            target_lang=target_lang if target_lang != "auto" else None,
            explain=include_reasoning
        )

        print(f"\n=== Legacy Model ===")
        legacy_result = legacy_pipeline.query(
            image,
            question,
            target_lang=target_lang if target_lang != "auto" else None
        )

        # Format output
        output = f"""
## 🔍 Question Analysis

**Original Question:** {modern_result['question_original']}
**English Translation:** {modern_result['question_english']}
**Detected Language:** {modern_result['source_lang']} → **Target Language:** {modern_result['target_lang']}

---

## 🤖 Model Comparison

### 🌟 Modern Model (LLaVA-1.6, 2024)

**Answer:** {modern_result.get('answer', 'Error: ' + modern_result.get('error', 'Unknown error'))}

{f"**English:** {modern_result.get('answer_english', 'N/A')}" if modern_result.get('answer_english') else ""}

---

### 📚 Legacy Model (ViLT, 2021)

**Answer:** {legacy_result.get('answer', 'Error: ' + legacy_result.get('error', 'Unknown error'))}

{f"**English:** {legacy_result.get('answer_english', 'N/A')}" if legacy_result.get('answer_english') else ""}

---

## 📊 Key Differences

- **Modern Model**: Provides detailed, contextual answers with reasoning
- **Legacy Model**: Gives concise, single-word/phrase answers
- **Translation Quality**: Both use Google Translate for multilingual support

"""

        return output

    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check that the models are loaded correctly."


# Create Gradio interface
demo = gr.Interface(
    fn=compare_vqa,
    inputs=[
        gr.Image(type="pil", label="📷 Upload Image"),
        gr.Textbox(
            label="❓ Your Question (in any language)",
            placeholder="Example: What are the cats doing? / ¿Qué están haciendo los gatos? / Que font les chats?",
            lines=2
        ),
        gr.Dropdown(
            choices=[
                ("Auto-detect", "auto"),
                ("English", "en"),
                ("Spanish", "es"),
                ("French", "fr"),
                ("German", "de"),
                ("Italian", "it"),
                ("Portuguese", "pt"),
                ("Chinese (Simplified)", "zh-cn"),
                ("Japanese", "ja"),
                ("Korean", "ko"),
                ("Arabic", "ar"),
                ("Russian", "ru"),
                ("Hindi", "hi"),
                ("Uyghur", "ug"),
                ("Kurdish (Kurmanji)", "ku"),
            ],
            label="🌍 Target Answer Language",
            value="auto"
        ),
        gr.Checkbox(
            label="💡 Include Reasoning (Modern Model Only)",
            value=True
        )
    ],
    outputs=gr.Markdown(label="Results"),
    title="🔍 Modern Cross-Lingual VQA",
    description="""
    ### Compare ViLT (2021) vs LLaVA-1.6 (2024) for Multilingual Visual Question Answering

    **Supported Languages:** 100+ languages via Google Translate
    **Models:** Modern (LLaVA-1.6-Mistral-7B) vs Legacy (ViLT-B32)

    Ask questions about images in any language and get answers in your preferred language!
    """,
    theme=gr.themes.Soft(),
    article="""
    ---
    ### 📖 About This Demo

    This demo showcases the evolution of Visual Question Answering (VQA) models:

    - **ViLT (2021)**: Lightweight vision-language transformer trained on VQAv2
    - **LLaVA-1.6 (2024)**: Advanced multimodal LLM with reasoning capabilities

    ### 🚀 Key Features

    - **Multilingual Support**: Ask questions in 100+ languages
    - **Cross-lingual QA**: Answer in a different language than the question
    - **Reasoning**: Modern models explain their answers
    - **Comparison**: See improvements from 2021 to 2024

    ### 🔗 Links

    - [GitHub Repository](https://github.com/AdivA-24/modern-vqa-extension)
    - [Original Paper](https://huggingface.co/spaces/ixxan/multilingual-vqa)
    - [Hugging Face Models](https://huggingface.co/models)

    ---

    **Note:** First load may take 1-2 minutes while models download.
    """
)

if __name__ == "__main__":
    demo.launch()
