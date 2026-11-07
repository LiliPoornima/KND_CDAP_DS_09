"""UI smoke tests: the page opens, every section renders, and a missing model shows a helpful message."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from backend.src.financial_capability import fc_predictor  # noqa: E402
from backend.src.financial_capability.fc_config import MODEL_DIR  # noqa: E402
from frontend.financial_capability import data_loader  # noqa: E402
from frontend.financial_capability.constants import SECTIONS  # noqa: E402

PAGE = str(Path(__file__).resolve().parents[2] / "frontend" / "pages" / "7_Financial_Capability.py")
TRAINED = fc_predictor.artifacts_available(MODEL_DIR)


def _page_text(at) -> str:
    parts = [m.value for m in at.markdown] + [w.value for w in at.warning] + \
            [i.value for i in at.info] + [c.value for c in at.code]
    return "\n".join(str(p) for p in parts)


def test_page_opens_and_lists_all_sections(monkeypatch):
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    assert list(at.radio[0].options) == SECTIONS


def test_missing_artifacts_show_not_trained_message(monkeypatch, tmp_path):
    monkeypatch.setattr(fc_predictor, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(data_loader, "MODEL_DIR", tmp_path)
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    for section in ("Model Performance", "Individual Assessment", "SHAP Explanation",
                    "Capability Gaps & Suggested Categories", "What-if Analysis"):
        at.radio[0].set_value(section).run()
        assert not at.exception, f"{section} crashed"
        text = _page_text(at)
        assert "not trained" in text.lower() or "No assessment yet" in text
    at.radio[0].set_value("Model Performance").run()
    assert "fc_run_pipeline" in _page_text(at)  # training command is shown


@pytest.mark.skipif(not TRAINED, reason="model not trained")
@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_renders_with_trained_model(section):
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    at.radio[0].set_value(section).run()
    assert not at.exception, f"{section}: {[e.value for e in at.exception]}"


@pytest.mark.skipif(not TRAINED, reason="model not trained")
def test_assessment_flow_end_to_end():
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    at.radio[0].set_value("Individual Assessment").run()
    next(b for b in at.button if b.label == "Use example answers").click().run()
    next(b for b in at.button if b.label == "Run assessment").click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Predicted financial well-being category"] in {"Low", "Medium", "High"}
    at.radio[0].set_value("SHAP Explanation").run()
    assert any("Why this prediction" in m.value for m in at.markdown)
    for section in ("Capability Gaps & Suggested Categories", "What-if Analysis"):
        at.radio[0].set_value(section).run()
        assert not at.exception, f"{section} crashed after an assessment"
