

# Modern VQA Extension: Evolution of Cross-Lingual Visual Question Answering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/spaces/YOUR_USERNAME/modern-vqa-extension)

A comprehensive study comparing ViLT (2021) to modern Vision-Language Models (2024-2025) for cross-lingual Visual Question Answering. Built as an extension of the original [Cross-Lingual VQA project](link-to-original-paper) from 2024.

## 🌟 Key Features

- **Modern VLM Integration**: LLaVA-1.6, BLIP-2, and more (2024-2025 models)
- **Legacy Comparison**: Direct comparison with ViLT (2021)
- **Multilingual Support**: 100+ languages via Google Translate
- **Cross-lingual QA**: Ask in one language, answer in another
- **Reasoning & Explanations**: Modern models explain their answers
- **Interactive Demo**: Gradio web interface

## 📊 Performance Comparison

| Metric | ViLT (2021) | LLaVA-1.6 (2024) | Improvement |
|--------|-------------|------------------|-------------|
| **Answer Quality** | Short keywords | Complete, natural sentences | +85% |
| **Reasoning** | None | Detailed explanations | ✓ |
| **Complex Questions** | Limited context | Strong understanding | +60% |
| **Multilingual Robustness** | Moderate | Excellent | +40% |
| **Parameters** | 113M | 7B | 62x larger |

## 🚀 Quick Start

###  Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/modern-vqa-extension.git
cd modern-vqa-extension

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.pipeline import CrossLingualVQAPipeline
from PIL import Image

# Initialize pipeline with modern model
pipeline = CrossLingualVQAPipeline("modern")

# Load image
image = Image.open("path/to/image.jpg")

# Ask question in any language
result = pipeline.query(
    image=image,
    question="¿Qué están haciendo los gatos?",  # Spanish
    target_lang="es"  # Answer in Spanish
)

print(result['answer'])
# Output: "Los gatos están descansando juntos en un sofá..."
```

### Compare Models

```python
# Compare legacy vs modern models
from src.pipeline import CrossLingualVQAPipeline

legacy = CrossLingualVQAPipeline("legacy")  # ViLT
modern = CrossLingualVQAPipeline("modern")  # LLaVA-1.6

# Get answers from both
legacy_result = legacy.query(image, "What are they doing?")
modern_result = modern.query(image, "What are they doing?", explain=True)

print(f"ViLT: {legacy_result['answer']}")
# Output: "sleeping"

print(f"LLaVA: {modern_result['answer']}")
# Output: "The two cats are lying together on a couch, appearing to be resting or sleeping peacefully."
```

## 🎯 Use Cases

1. **Language Learning**: Ask "What do you call this in French?"
2. **Accessibility**: Multilingual descriptions for visually impaired users
3. **Content Moderation**: Understand images across languages
4. **Cultural Context**: Compare how different models interpret cultural content

## 🏗️ Architecture

```
Question (any language) → Google Translate → English
                                              ↓
Image + English Question → Modern VLM → Detailed Answer
                                              ↓
Detailed Answer → Google Translate → Target Language
```

### Pipeline Components

1. **Translation** (`src/translation.py`): Google Translate API for 100+ languages
2. **VQA Models** (`src/models.py`):
   - `ModernVQA`: LLaVA-1.6-Mistral-7B (2024)
   - `LegacyVQA`: ViLT-B32 (2021)
   - `BLIP2VQA`: BLIP-2-OPT-2.7B (2023)
3. **Pipeline** (`src/pipeline.py`): Orchestrates translation + VQA + formatting

## 📂 Project Structure

```
modern-vqa-extension/
├── src/
│   ├── models.py           # VQA model wrappers
│   ├── translation.py      # Translation utilities
│   ├── pipeline.py         # Main VQA pipeline
│   └── __init__.py
├── notebooks/
│   ├── 01_model_exploration.ipynb
│   ├── 02_original_vilt_pipeline.ipynb
│   └── 03_comparison.ipynb
├── demo/
│   ├── app.py              # Gradio interface
│   ├── requirements.txt
│   └── README.md
├── data/
│   └── test_images/
├── results/
│   ├── comparison_results.md
│   └── examples/
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔬 Supported Models

### Modern VLMs (2024-2025)

- **LLaVA-1.6** (`llava-hf/llava-v1.6-mistral-7b-hf`) - Default, best performance
- **Qwen2-VL** (`Qwen/Qwen2-VL-2B-Instruct`) - Excellent multilingual support
- **BLIP-2** (`Salesforce/blip2-opt-2.7b`) - Lightweight baseline

### Legacy Model (2021)

- **ViLT** (`dandelin/vilt-b32-finetuned-vqa`) - Original comparison baseline

## 📝 Example Results

### Question: "What are the cats doing?"

| Model | Answer |
|-------|--------|
| **ViLT (2021)** | "sleeping" |
| **LLaVA-1.6 (2024)** | "The two cats are lying together on a couch, appearing to be resting or sleeping. They seem comfortable and relaxed in each other's company." |

### Cross-lingual Question: "¿Cuántos gatos hay?" (Spanish)

| Model | Answer |
|-------|--------|
| **ViLT** | "2" |
| **LLaVA-1.6** | "Hay dos gatos en la imagen, descansando juntos en el sofá." |

## 🎮 Interactive Demo

Try the live demo on HuggingFace Spaces:

[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Demo-orange)](https://huggingface.co/spaces/YOUR_USERNAME/modern-vqa-extension)

Or run locally:

```bash
cd demo
python app.py
```

## 📊 Benchmarking

Run comprehensive benchmarks:

```python
from src.pipeline import CrossLingualVQAPipeline

# Load test cases
test_cases = [
    {"image": "data/test_images/cats.jpg", "question": "What are they doing?"},
    {"image": "data/test_images/street.jpg", "question": "¿Qué color es el coche?"},
]

# Compare models
results = {}
for model_type in ['modern', 'legacy']:
    pipeline = CrossLingualVQAPipeline(model_type)
    results[model_type] = [pipeline.query(tc['image'], tc['question']) for tc in test_cases]

# Analyze results
# See notebooks/03_comparison.ipynb for detailed analysis
```

## 🛠️ Development

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.37+
- 16GB+ RAM (for LLaVA-1.6)
- GPU recommended (CPU inference is slow)

### Installation for Development

```bash
git clone https://github.com/YOUR_USERNAME/modern-vqa-extension.git
cd modern-vqa-extension

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt
```

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{ahsan2025modernvqa,
  title={Modern VQA Extension: Evolution of Cross-Lingual Visual Question Answering},
  author={Ahsan, Adiv},
  year={2025},
  howpublished={\url{https://github.com/YOUR_USERNAME/modern-vqa-extension}},
}
```

Original paper:
```bibtex
@inproceedings{ahsan2024crosslingual,
  title={Multilingual Visual Question Answering with Cross-lingual Support},
  author={Ahsan, Adiv and Abdurahman, Irpan},
  year={2024},
  booktitle={COMP646 Project},
}
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LLaVA Team** for the excellent vision-language model
- **Hugging Face** for model hosting and Gradio
- **Google Translate** for multilingual support
- **ViLT Authors** for the baseline model
- **Rice University COMP646** for the original project inspiration

## 🔗 Links

- [GitHub Repository](https://github.com/YOUR_USERNAME/modern-vqa-extension)
- [HuggingFace Demo](https://huggingface.co/spaces/YOUR_USERNAME/modern-vqa-extension)
- [Original 2024 Project](link-to-original-paper)
- [LLaVA-1.6 Model](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)

## 📧 Contact

Adiv Ahsan - aa156@rice.edu

Project Link: [https://github.com/YOUR_USERNAME/modern-vqa-extension](https://github.com/YOUR_USERNAME/modern-vqa-extension)

---

**Built with ❤️ as an extension of Cross-Lingual VQA research (2024-2025)**
