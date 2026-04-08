# AIMO-Marks-V1 | Mathematical Reasoning Engine

Fine-tuned DeepSeek-Math-7B model for step-by-step mathematical reasoning, optimized for 4-bit inference on consumer GPUs.

## 🎯 Overview

AIMO-Marks-V1 is a specialized math reasoning model built by fine-tuning DeepSeek-Math-7B-RL using LoRA (Low-Rank Adaptation). The model excels at solving algebra, arithmetic, and competition-style math problems with clear step-by-step reasoning.

## ✨ Features

- **Step-by-step reasoning** - Provides clear, logical derivations
- **4-bit quantization** - Runs on GPUs with as little as 6GB VRAM
- **LoRA fine-tuned** - Efficient adaptation for math problems
- **Gradio interface** - Easy-to-use web UI for interaction
- **Memory optimized** - Includes VRAM cleanup utilities

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (6GB+ VRAM recommended)
- 8GB+ system RAM

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/JoshuaOmosa/aimo-marks-v1.git
cd aimo-marks-v1

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

pip install -r requirements.txt

marks_v1_model/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
└── tokenizer_config.json

Run the demo
bash
python demo_app.py

Usage Examples
Problem: Bat and Ball
Input:

A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?

Output:

text
Let x be the cost of the ball in dollars.
Then the cost of the bat is x + 1.00.
Total: x + (x + 1.00) = 1.10
2x + 1.00 = 1.10
2x = 0.10
x = 0.05

Therefore, the ball costs $0.05 (5 cents).
Problem: Prime Numbers
Input:

What is the sum of all prime numbers less than 20?

Output:

text
Prime numbers less than 20: 2, 3, 5, 7, 11, 13, 17, 19
Sum = 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77

 Project Structure
text
aimo-marks-v1/
├── demo_app.py              # Gradio web interface
├── train_aimo.py            # LoRA fine-tuning script
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # Documentation
├── marks_v1_model/         # Trained adapter weights
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── utils/                  # Utility functions
│   ├── memory_utils.py     # GPU memory management
│   └── system_cleaner.py   # System resource cleanup
├── checkpoints/            # Training checkpoints (created during training)
└── curated_data/          # Processed dataset (created during curation)
 Training from Scratch

1. Curate dataset (problems with 5-digit answers)
bash
python -c "from curation_pipeline import run_curation; run_curation()"

2. Train the model
bash
python train_aimo.py

3. Export trained model
bash
python scripts/export_model.py

Memory Requirements
Component	VRAM Usage
Base Model (4-bit)	~4-5 GB
LoRA Adapter	~0.1 GB
Gradients (training)	~2-3 GB
Inference Total	~5-6 GB
Training Total	~8-10 GB
Training Details
Parameter	Value
Base Model	DeepSeek-Math-7B-RL
Fine-tuning	LoRA (r=8, alpha=16)
Quantization	4-bit NF4 + double quantization
Optimizer	Paged AdamW 8-bit
Learning Rate	2e-4
Batch Size	1 (4 gradient accumulation)
Dataset	AI-MO/NuminaMath-CoT (5-digit filtered)
Training Time	~2-3 hours on dual T4 GPUs

Troubleshooting
Out of Memory (OOM)
Reduce max_new_tokens in demo_app.py:

python
max_new_tokens=256  # Instead of 512
Adapter not found
Ensure marks_v1_model/ contains:

adapter_model.safetensors

adapter_config.json

Port already in use
Change port in demo_app.py:

python
demo.launch(share=True, server_port=7861)
📝 License
MIT License - See LICENSE file

Acknowledgments

DeepSeek AI for the base model

AI-MO for NuminaMath-CoT dataset

HuggingFace for transformers library


