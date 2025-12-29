# ProfASR-Bench

**A Professional-Talk ASR Benchmark for High-Stakes Applications**

[![Dataset on HuggingFace](https://img.shields.io/badge/🤗%20Dataset-ProfASR--Bench-blue)](https://huggingface.co/datasets/prdeepakbabu/ProfASR-Bench)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

## 🎯 Overview

ProfASR-Bench is a professional-talk evaluation suite for **context-conditioned ASR** in high-stakes applications. It exposes the **Context-Utilization Gap (CUG)** - the phenomenon where modern ASR systems are nominally promptable yet underuse readily available side information.

### Key Features

- **3,200 utterances** across 4 professional domains (Finance, Medicine, Legal, Technology)
- **4 voice profiles** (2 American, 2 British; 2 male, 2 female)
- **Entity-rich content** with typed named entities for NE-WER evaluation
- **Context Ladder protocol** for systematic prompt conditioning experiments
- **High-quality synthetic speech** via Kokoro 82M TTS (Apache 2.0)

## 📊 Key Finding: The Context-Utilization Gap

| Model | NO-PROMPT WER | +PROFILE | +DOMAIN | ORACLE | Gap |
|-------|---------------|----------|---------|--------|-----|
| Whisper-large-v3 | 5.2% | 5.1% | 4.9% | 3.8% | 1.4% |
| Whisper-turbo | 6.8% | 6.7% | 6.5% | 4.2% | 2.6% |
| Assembly AI | 4.1% | 4.0% | 3.9% | 3.2% | 0.9% |

*Modern systems show minimal improvement with context—far below the ORACLE ceiling.*

## 🚀 Quick Start

### Load Dataset

```python
from datasets import load_dataset

# Load from HuggingFace
dataset = load_dataset("prdeepakbabu/ProfASR-Bench")

# Access samples
for sample in dataset["train"]:
    audio = sample["audio"]
    truth = sample["truth"]      # Ground truth transcription
    prompt = sample["prompt"]    # Context sentences
    domain = sample["domain"]    # FINANCIAL, MEDICAL, LEGAL, TECHNICAL
```

### Run Evaluation

```python
from evaluation.metrics import compute_wer, compute_ner_wer

# Standard WER
wer = compute_wer(predictions, references)

# Entity-aware NE-WER  
ne_wer, entity_f1 = compute_ner_wer(predictions, references, named_entities)
```

## 📁 Repository Structure

```
ProfASR-Bench/
├── data_generation/           # Dataset creation pipeline
│   ├── text/                  # Text generation (Claude prompts)
│   │   ├── domains.py         # Domain definitions
│   │   ├── utterance_generator.py  # LLM prompt templates
│   │   └── profile_generator.py    # Speaker profile creation
│   └── audio/                 # TTS synthesis (Kokoro 82M)
│       ├── kokoro_tts_generator.py
│       └── batch_processor.py
│
├── evaluation/                # ASR evaluation code
│   ├── metrics.py             # WER, NE-WER, Entity-F1
│   ├── asr_models.py          # Whisper wrappers
│   └── data_loader.py         # Dataset loading utilities
│
├── configs/                   # Configuration files
│   ├── prompt_configs.py      # Context Ladder prompts
│   └── model_configs.py       # Model settings
│
└── notebooks/                 # Demo notebooks
    └── whisper_evaluation.ipynb
```

## 🔬 Evaluation Protocol: Context Ladder

Test ASR systems across 5 prompt conditions:

| Condition | Description |
|-----------|-------------|
| **NO-PROMPT** | Control baseline - no context |
| **PROFILE** | Speaker attributes only ("mid-thirties analyst from Toronto") |
| **DOMAIN+PROFILE** | Domain cue + speaker attributes |
| **ORACLE** | Gold transcript as prompt (ceiling reference) |
| **ADVERSARIAL** | Mismatched domain prompt (robustness test) |

## 📈 Metrics

- **WER**: Word Error Rate (standard)
- **NE-WER**: Named Entity WER (entity-weighted)
- **Entity-F1**: Precision/Recall on domain entities
- **Slice Analysis**: Accent gaps (American vs British), Gender gaps

## 🔧 Installation

```bash
git clone https://github.com/prdeepakbabu/ProfASR-Bench.git
cd ProfASR-Bench
pip install -r requirements.txt
```

## 📖 Citation

```bibtex
@article{piskala2025profasrbench,
  title={ProfASR-Bench: A Professional-Talk ASR Dataset for High-Stakes Applications Exposing the Context-Utilization Gap},
  author={Piskala, Deepak Babu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  url={https://arxiv.org/abs/XXXX.XXXXX}
}
```

## 📜 License

Apache 2.0 License. The synthetic audio was generated using [Kokoro 82M TTS](https://github.com/hexgrad/kokoro) (permissive licensing).

## 🔗 Links

- **Dataset**: [HuggingFace Hub](https://huggingface.co/datasets/prdeepakbabu/ProfASR-Bench)
- **Paper**: [arXiv](https://arxiv.org/abs/XXXX.XXXXX)
- **Author**: [Deepak Babu Piskala](https://prdeepakbabu.github.io/)
