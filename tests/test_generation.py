from types import SimpleNamespace

from PIL import Image

from app import generation
from app.config import Settings


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


def _run(monkeypatch, turbo, hires):
    calls = []
    pipes = {"text": FakePipe("text", calls), "img2img": FakePipe("img2img", calls)}
    monkeypatch.setattr(generation, "get_pipeline", pipes.__getitem__)
    monkeypatch.setattr(generation, "get_loaded_adapters", lambda: [])
    settings = Settings(hires_fix=hires)
    monkeypatch.setattr(generation, "get_settings", lambda: settings)
    image = generation.generate_image("text", "Homer", steps=30, turbo=turbo)
    return image, calls


def test_hires_fix_upscales_then_refines(monkeypatch):
    image, calls = _run(monkeypatch, turbo=False, hires=True)
    assert [c[0] for c in calls] == ["text", "img2img"]
    assert calls[1][1]["strength"] == 0.4 and image.size == (768, 768)


def test_hires_fix_skipped_for_turbo_or_when_disabled(monkeypatch):
    assert [c[0] for c in _run(monkeypatch, turbo=True, hires=True)[1]] == ["text"]
    assert [c[0] for c in _run(monkeypatch, turbo=False, hires=False)[1]] == ["text"]
