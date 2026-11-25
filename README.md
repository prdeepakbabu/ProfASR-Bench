# ProfASR-Bench

**A Professional-Talk ASR Benchmark for High-Stakes Applications**

[![Dataset on HuggingFace](https://img.shields.io/badge/🤗%20Dataset-ProfASR--Bench-blue)](https://huggingface.co/datasets/prdeepakbabu/ProfASR-Bench)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2512.23686-b31b1b.svg)](https://arxiv.org/abs/2512.23686)

## 🎯 Overview

ProfASR-Bench is a professional-talk evaluation suite for **context-conditioned ASR** in high-stakes applications. It exposes the **Context-Utilization Gap (CUG)** - the phenomenon where modern ASR systems are nominally promptable yet underuse readily available side information.

### Key Features

- **3,200 utterances** across 4 professional domains (Finance, Medicine, Legal, Technology)
- **4 voice profiles** (2 American, 2 British; 2 male, 2 female)
- **Entity-rich content** with typed named entities for NE-WER evaluation
- **Context Ladder protocol** for systematic prompt conditioning experiments
- **High-quality synthetic speech** via Kokoro 82M TTS (Apache 2.0)

## 📊 Key Finding: The Context-Utilization Gap

### Overall WER by Model (No Context)

| Model | Overall | Financial | Legal | Medical | Technical |
|-------|---------|-----------|-------|---------|-----------|
| **Whisper Small** | **10.0%** | 13.3% | 8.5% | 15.8% | **2.3%** |
| Whisper Base | 12.1% | 14.6% | 11.1% | 17.9% | 4.7% |
| Whisper Tiny | 14.3% | 15.8% | 13.8% | 21.4% | 6.3% |
| Qwen 2.5 Omni 3B | 24.3% | 15.2% | 35.7% | 38.9% | 7.3% |

### Context Ladder Results (Whisper Small)

| Condition | WER | SER | ΔWER vs No-prompt |
|-----------|-----|-----|-------------------|
| No-prompt | 9.98% | 52.56% | — |
| Profile | 9.95% | 52.44% | −0.03 pp |
| Domain+Profile | 9.95% | 52.38% | −0.03 pp |
| **Oracle** | **9.92%** | 52.44% | −0.06 pp |
| Adversarial | 9.95% | 52.50% | −0.03 pp |

*Lightweight textual context yields little to no change in average WER—even at an ORACLE ceiling—exposing the Context-Utilization Gap (CUG).*

### Additional Evaluation: Oruk Fourier 2B (`oruk/oruk-fourier-2b`)

Evaluated on `prdeepakbabu/ProfASR-Bench@ebb55ae0` (3,200 utterances, 8.59 h; NVIDIA H200 BF16, `transformers==5.24.0`, pinned `oruk/oruk-fourier-2b@8c2a8871`, CC BY-NC 4.0 weights). Note: 160/3,200 dataset rows (`68` financial, `24` legal, `68` medical, `0` technical) contain punctuation-only `truth` (`".`), which normalize to empty strings; both corpus micro-WER over the 3,040 non-empty utterances (59,184 words) and Whisper `EnglishTextNormalizer` corpus WER (60,724 words) are reported alongside the raw all-3,200 `WERCalculator` mean.

| Decoding Mode | ProfASR Corpus WER (3,040 valid) | Financial | Legal | Medical | Technical | Whisper-Norm Corpus WER | Raw All-3,200 Mean WER | Throughput |
|---|---|---|---|---|---|---|---|---|
| **3-beam (`published`, default)** | **6.94%** | **8.62%** | **3.46%** | **6.23%** | **9.55%** | **5.07%** | 12.17% | 302× RTFx |
| Greedy (`num_beams=1`) | 7.38% | 8.77% | 3.99% | 6.56% | 10.27% | 5.51% | 12.60% | 666× RTFx |

| Context Ladder Condition (`oruk/oruk-fourier-2b`, 3-beam) | ProfASR Corpus WER | Δ vs No-prompt | Whisper-Norm WER | Δ vs No-prompt | Valid SER |
|---|---|---|---|---|---|
| No-prompt | 6.94% | — | 5.07% | — | 56.28% |
| **Profile** | **6.84%** | −0.10 pp | **5.01%** | −0.06 pp | 55.89% |
| Domain+Profile | 7.03% | +0.09 pp | 5.11% | +0.04 pp | 57.40% |
| Dataset Prompt | 6.82% | −0.12 pp | 5.08% | +0.01 pp | 55.53% |
| Oracle | 14.49% | +7.55 pp | 12.15% | +7.08 pp | 88.39% |
| Adversarial | 7.66% | +0.72 pp | 5.76% | +0.69 pp | 60.56% |

*Even on a 2B Qwen3-ASR model with native prompt conditioning, speaker/domain cues yield ≤0.12 pp WER gain while mismatched (`Adversarial`) or full-transcript (`Oracle`) prompts degrade accuracy—reinforcing the Context-Utilization Gap.*

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
  journal={arXiv preprint arXiv:2512.23686},
  year={2025},
  url={https://arxiv.org/abs/2512.23686}
}
```

## 📜 License

Apache 2.0 License. The synthetic audio was generated using [Kokoro 82M TTS](https://github.com/hexgrad/kokoro) (permissive licensing).

## 🔗 Links

- **Dataset**: [HuggingFace Hub](https://huggingface.co/datasets/prdeepakbabu/ProfASR-Bench)
- **Paper**: [arXiv](https://arxiv.org/abs/2512.23686)
- **Author**: [Deepak Babu Piskala](https://prdeepakbabu.github.io/)
