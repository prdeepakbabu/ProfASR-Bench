"""Unit tests for Oruk Fourier 2B config and Qwen3ASRModelWrapper in ProfASR-Bench."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from configs.model_configs import get_model_config, list_available_models
from configs.prompt_configs import get_prompt_config
from evaluation.asr_models import Qwen3ASRModelWrapper


def test_oruk_fourier_config_registered_and_pinned():
    assert "oruk-fourier-2b" in list_available_models()
    cfg = get_model_config("oruk-fourier-2b")
    assert cfg.model_id == "oruk/oruk-fourier-2b"
    assert cfg.revision == "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
    assert cfg.architecture == "qwen3_asr"


def test_qwen3_wrapper_decode_modes_and_revision_pin():
    cfg = get_model_config("oruk-fourier-2b")
    cfg.device = "cpu"
    cfg.torch_dtype = torch.float32

    pub = Qwen3ASRModelWrapper(cfg, decode_mode="published")
    assert pub._generation_kwargs() == {}

    greedy = Qwen3ASRModelWrapper(cfg, decode_mode="greedy")
    assert greedy._generation_kwargs(max_new_tokens=256) == {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 256,
    }

    with pytest.raises(ValueError):
        Qwen3ASRModelWrapper(cfg, decode_mode="invalid")


def test_qwen3_wrapper_loads_with_pinned_revision_and_passes_prompts():
    cfg = get_model_config("oruk-fourier-2b")
    cfg.device = "cpu"
    cfg.torch_dtype = torch.float32

    captured_req = {}

    class _FakeBatch(dict):
        def to(self, device, dtype=None):
            return self

    fake_proc = MagicMock()
    fake_proc.tokenizer = MagicMock()

    def _apply_req(**kwargs):
        captured_req.update(kwargs)
        return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]])})

    fake_proc.apply_transcription_request.side_effect = _apply_req
    fake_proc.decode.return_value = ["EBITDA margins rose.", "Prescribe azithromycin."]

    fake_model = MagicMock()
    fake_model.device = torch.device("cpu")
    fake_model.dtype = torch.float32
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.generate.return_value = torch.tensor([[1, 2, 3, 10, 11], [1, 2, 3, 12, 13]])

    wrapper = Qwen3ASRModelWrapper(cfg, decode_mode="published")
    with (
        patch("evaluation.asr_models.AutoProcessor.from_pretrained", return_value=fake_proc) as p_proc,
        patch.dict(
            "sys.modules",
            {
                "transformers": types.SimpleNamespace(
                    AutoModelForMultimodalLM=MagicMock(
                        from_pretrained=MagicMock(return_value=fake_model)
                    )
                )
            },
        ),
    ):
        wrapper.load_model()
        p_proc.assert_called_once_with(
            "oruk/oruk-fourier-2b",
            revision="8c2a8871f1304aa2f634da4fb95e24438c4945b9",
        )

    out = wrapper.transcribe_batch_with_prompts(
        audio_arrays=[np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.float32)],
        sample_metadata_list=[
            {"domain": "financial", "voice": "heart"},
            {"domain": "medical", "voice": "george"},
        ],
        prompt_config=get_prompt_config("domain_only"),
    )
    assert out["predictions"] == ["EBITDA margins rose.", "Prescribe azithromycin."]
    assert len(captured_req["prompt"]) == 2
    assert captured_req["language"] == ["English", "English"]


def test_wer_calculator_detailed_jiwer_compat():
    from evaluation.metrics import WERCalculator

    calc = WERCalculator()
    wer, details = calc.calculate_wer("Take 50mg daily.", "Take 50mg twice daily.", return_details=True)
    assert details.reference_length == 3
    assert details.insertions == 1
    assert details.total_errors == 1
    assert wer == pytest.approx(1.0 / 3.0)
