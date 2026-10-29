"""Abstraction de stockage des images.

L'ancienne API écrivait directement sur le disque du process qui sert aussi le
HTTP (`generated_simpsons/`), un dossier qui a fini par être versionné dans
Git. Ici, le disque local reste le backend par défaut pour le dev, mais un
backend S3 (AWS S3, Cloudflare R2, ou tout endpoint compatible) est disponible
en réglant STORAGE_BACKEND=s3 (+ S3_BUCKET et S3_PUBLIC_BASE_URL).
"""

from __future__ import annotations

import abc
from functools import lru_cache
from pathlib import Path

from .config import get_settings


class StorageBackend(abc.ABC):
    @abc.abstractmethod
    def save(self, data: bytes, key: str) -> None: ...

    @abc.abstractmethod
    def load(self, key: str) -> bytes: ...

    @abc.abstractmethod
    def url_for(self, key: str) -> str: ...

    @abc.abstractmethod
    def delete(self, key: str) -> None: ...


class LocalStorage(StorageBackend):
    def __init__(self, root: Path, public_base_url: str) -> None:
        self.root = root
        self.public_base_url = public_base_url.rstrip("/")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError(f"Clé de stockage invalide : {key!r}")
        return path

    def save(self, data: bytes, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def load(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def url_for(self, key: str) -> str:
        return f"{self.public_base_url}/images/{key}"


class S3Storage(StorageBackend):
    def __init__(
        self,
        bucket: str,
        region: str | None,
        endpoint_url: str | None,
        public_base_url: str,
        prefix: str = "",
    ) -> None:
        import boto3  # import différé : pas nécessaire tant que STORAGE_BACKEND != "s3"

        self.client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
        self.bucket = bucket
        self.prefix = prefix
        self.public_base_url = public_base_url.rstrip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def save(self, data: bytes, key: str) -> None:
        content_type = "image/png" if key.endswith(".png") else "application/octet-stream"
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, ContentType=content_type)

    def load(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        return obj["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def url_for(self, key: str) -> str:
        return f"{self.public_base_url}/{self._key(key)}"


@lru_cache
def get_storage() -> StorageBackend:
    settings = get_settings()
    if settings.storage_backend == "s3":
        if not settings.s3_bucket or not settings.s3_public_base_url:
            # Pas d'URL signée : elle serait stockée en base et expirerait (galerie cassée après 1 h).
            raise RuntimeError("STORAGE_BACKEND=s3 nécessite S3_BUCKET et S3_PUBLIC_BASE_URL.")
        return S3Storage(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            public_base_url=settings.s3_public_base_url,
            prefix=settings.s3_prefix,
        )
    return LocalStorage(root=settings.output_folder, public_base_url=settings.public_base_url)
