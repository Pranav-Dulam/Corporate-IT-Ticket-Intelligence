"""Model loading - spec cases: missing model, corrupted model."""

import pytest


def test_missing_model_degrades_without_raising(monkeypatch, tmp_path):
    import predictor

    monkeypatch.setattr(predictor, "MODEL_DIR", tmp_path)

    result = predictor.Predictor().predict("vpn keeps dropping")

    assert result["category"]["label"] is None
    assert result["category"]["uncertain"] is True
    assert result["model_errors"]


def test_corrupted_model_degrades_without_raising(monkeypatch, tmp_path):
    import predictor

    bad = tmp_path / predictor.CATEGORY_MODELS[0]
    bad.write_bytes(b"this is not a joblib file")

    monkeypatch.setattr(predictor, "MODEL_DIR", tmp_path)

    result = predictor.Predictor().predict("vpn keeps dropping")

    assert result["category"]["label"] is None
    assert any("failed to load" in e for e in result["model_errors"])


def test_binary_confidence_floor_is_above_chance():
    """
    A 2-class model at 54% is a coin flip. A shared floor of 0.35 could
    never fire for it, so a coin flip was being shown as confident.
    """
    import predictor

    assert predictor.CONFIDENCE_FLOORS["priority"] > 0.5
    assert predictor.CONFIDENCE_FLOORS["category"] < 0.5