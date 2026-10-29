import pytest

from app import storage
from app.config import Settings


def test_s3_requires_public_base_url(monkeypatch):
    monkeypatch.setattr(storage, "get_settings", lambda: Settings(storage_backend="s3", s3_bucket="b"))
    with pytest.raises(RuntimeError, match="S3_PUBLIC_BASE_URL"):
        storage.get_storage.__wrapped__()
