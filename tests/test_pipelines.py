import numpy as np
from safetensors.numpy import save_file

from app.pipelines import _lora_matches_unet


def _lora(tmp_path, key, shape):
    path = tmp_path / "lora.safetensors"
    save_file({key: np.zeros(shape, dtype=np.float16)}, str(path))
    return path


def test_lora_accepted_when_cross_attention_dim_matches(tmp_path):
    for key in ("unet.x.attn2.to_k.lora_A.weight", "unet.x.attn2.to_k.lora.down.weight"):  # peft, ancien format
        assert _lora_matches_unet(_lora(tmp_path, key, (4, 768)), 768)


def test_lora_refused_for_other_base_or_unknown_format(tmp_path):
    sdxl = _lora(tmp_path, "unet.x.attn2.to_k.lora_A.weight", (4, 2048))
    assert not _lora_matches_unet(sdxl, 768)
    unknown = _lora(tmp_path, "unet.x.attn2.to_k.weight", (4, 768))
    assert not _lora_matches_unet(unknown, 768)
