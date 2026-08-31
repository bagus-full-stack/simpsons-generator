"""Abstraction de stockage des images.

L'ancienne API écrivait directement sur le disque du process qui sert aussi le
HTTP (`generated_simpsons/`), un dossier qui a fini par être versionné dans
Git. Ici, le disque local reste le backend par défaut pour le dev, mais un
backend S3 (AWS S3, Cloudflare R2, ou tout endpoint compatible) est disponible
en changeant une seule variable d'environnement : STORAGE_BACKEND=s3.
"""

from __future__ import annotations

import abc
from pathlib import Path

from .config import get_settings


class StorageBackend(abc.ABC):
    @abc.abstractmethod
    def save(self, data: bytes, key: str) -> None: ...

    @abc.abstractmethod
    def load(self, key: str) -> bytes: ...

    @abc.abstractmethod
    def url_for(self, key: str) -> str: ...


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

    def url_for(self, key: str) -> str:
        return f"{self.public_base_url}/images/{key}"


class S3Storage(StorageBackend):
    def __init__(
        self,
        bucket: str,
        region: str | None,
        endpoint_url: str | None,
        public_base_url: str | None,
        prefix: str = "",
    ) -> None:
        import boto3  # import différé : pas nécessaire tant que STORAGE_BACKEND != "s3"

        self.client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
        self.bucket = bucket
        self.prefix = prefix
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def save(self, data: bytes, key: str) -> None:
        content_type = "image/png" if key.endswith(".png") else "application/octet-stream"
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, ContentType=content_type)

    def load(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        return obj["Body"].read()

    def url_for(self, key: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url}/{self._key(key)}"
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": self._key(key)}, ExpiresIn=3600
        )


def get_storage() -> StorageBackend:
    settings = get_settings()
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError("STORAGE_BACKEND=s3 nécessite S3_BUCKET.")
        return S3Storage(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            public_base_url=settings.s3_public_base_url,
            prefix=settings.s3_prefix,
        )
    return LocalStorage(root=settings.output_folder, public_base_url=settings.public_base_url)
