from app.config import Settings


def test_csv_fields_are_split_and_trimmed():
    settings = Settings(
        cors_allow_origins="http://a.com, http://b.com",
        api_keys="key-1, key-2",
        moderation_blocklist="x, y ,z",
    )
    assert settings.cors_allow_origins == ["http://a.com", "http://b.com"]
    assert settings.api_keys == ["key-1", "key-2"]
    assert settings.moderation_blocklist == ["x", "y", "z"]


def test_defaults_are_dev_friendly():
    settings = Settings()
    assert settings.environment == "development"
    assert settings.queue_backend == "inline"
    assert settings.storage_backend == "local"


def test_defaults_target_sdxl():
    settings = Settings()
    assert settings.base_model_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert settings.lcm_lora_id == "latent-consistency/lcm-lora-sdxl"
    assert settings.controlnet_model_id == "diffusers/controlnet-canny-sdxl-1.0"
    assert settings.image_resolution == 1024
