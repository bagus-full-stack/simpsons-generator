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


def test_defaults_target_sd15():
    settings = Settings()
    assert settings.base_model_id == "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    assert settings.lcm_lora_id == "latent-consistency/lcm-lora-sdv1-5"
    assert settings.controlnet_model_id == "lllyasviel/control_v11p_sd15_canny"
    assert settings.image_resolution == 512
    assert settings.lora_scale == 0.6
