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
