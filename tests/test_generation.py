import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app import generation
from app.config import Settings, get_settings


class FakePipe:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls
        self.scheduler = self.scheduler_default = self.scheduler_lcm = object()

    def __call__(self, **kwargs):
        self.calls.append((self.name, kwargs))
        size = kwargs["image"].size if "image" in kwargs else (512, 512)
        return SimpleNamespace(images=[Image.new("RGB", size)])

    def set_adapters(self, *a, **k):
        pass


def _run(monkeypatch, turbo, hires, steps=30):
    calls = []
    pipes = {"text": FakePipe("text", calls), "img2img": FakePipe("img2img", calls)}
    monkeypatch.setattr(generation, "get_pipeline", pipes.__getitem__)
    monkeypatch.setattr(generation, "LOADED_ADAPTERS", [])
    settings = Settings(hires_fix=hires)
    monkeypatch.setattr(generation, "get_settings", lambda: settings)
    image = generation.generate_image("text", "Homer", steps=steps, turbo=turbo)
    return image, calls


def test_hires_fix_upscales_then_refines(monkeypatch):
    image, calls = _run(monkeypatch, turbo=False, hires=True)
    assert [c[0] for c in calls] == ["text", "img2img"]
    assert calls[1][1]["strength"] == 0.4 and image.size == (768, 768)


def test_hires_fix_skipped_for_turbo_or_when_disabled(monkeypatch):
    assert [c[0] for c in _run(monkeypatch, turbo=True, hires=True)[1]] == ["text"]
    assert [c[0] for c in _run(monkeypatch, turbo=False, hires=False)[1]] == ["text"]


def test_hires_fix_keeps_at_least_one_step(monkeypatch):
    calls = _run(monkeypatch, turbo=False, hires=True, steps=2)[1]
    assert int(calls[1][1]["num_inference_steps"] * calls[1][1]["strength"]) >= 1


@pytest.mark.parametrize(
    "turbo, steps, strength", [(True, 30, 0.1), (False, 49, 0.02), (False, 1, 0.5), (True, 30, 0.0)]
)
def test_img2img_strength_never_means_zero_steps(monkeypatch, turbo, steps, strength):
    calls = []
    monkeypatch.setattr(generation, "get_pipeline", {"img2img": FakePipe("img2img", calls)}.__getitem__)
    monkeypatch.setattr(generation, "LOADED_ADAPTERS", [])
    monkeypatch.setattr(generation, "get_settings", lambda: Settings())
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="PNG")
    generation.generate_image(
        "img2img", "Homer", steps=steps, turbo=turbo, input_image_bytes=buf.getvalue(), strength=strength
    )
    args = calls[0][1]
    assert int(args["num_inference_steps"] * args["strength"]) >= 1


def test_warm_up_runs_text_then_img2img_unless_models_are_skipped(monkeypatch):
    calls = []
    monkeypatch.setattr(generation, "generate_image", lambda mode, *a, **k: calls.append(mode))
    settings = get_settings()
    monkeypatch.setattr(settings, "skip_model_load", True)
    generation.warm_up_pipelines()
    assert calls == []
    monkeypatch.setattr(settings, "skip_model_load", False)
    generation.warm_up_pipelines()
    assert calls == ["text", "img2img"]
