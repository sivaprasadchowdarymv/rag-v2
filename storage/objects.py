"""
Object storage for uploaded originals (versioning, re-indexing, backup).

    STORAGE_BACKEND=local   data/objects/<tenant>/<hash>        (default)
    STORAGE_BACKEND=s3      MinIO / AWS S3 / any S3-compatible store (needs boto3)

Keys are content hashes, so identical uploads are stored once per tenant.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from config.settings import Settings, get_logger

log = get_logger("objects")
_KEY = re.compile(r"^[A-Za-z0-9_\-]{1,64}/[A-Za-z0-9_.\-]{1,128}$")


def object_key(tenant_id: str, content_hash: str, filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")[:8]
    safe_tenant = re.sub(r"[^A-Za-z0-9_\-]", "_", tenant_id)[:64] or "default"
    return f"{safe_tenant}/{content_hash}{'.' + ext if ext else ''}"


def _check(key: str) -> str:
    if not _KEY.match(key) or ".." in key:
        raise ValueError("invalid object key")
    return key


class LocalObjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / _check(key)

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        return p.read_bytes() if p.exists() else None

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def list(self, prefix: str = "") -> List[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")
                      if p.is_file() and str(p.relative_to(self.root)).startswith(prefix))


class S3ObjectStore:
    def __init__(self, settings: Settings):
        import boto3  # optional dependency

        self.bucket = settings.s3_bucket
        self.client = boto3.client("s3", endpoint_url=settings.s3_endpoint or None,
                                   aws_access_key_id=settings.s3_access_key or None,
                                   aws_secret_access_key=settings.s3_secret_key or None)
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=_check(key), Body=data)

    def get(self, key: str) -> Optional[bytes]:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=_check(key))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=_check(key))

    def list(self, prefix: str = "") -> List[str]:
        out, token = [], None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix, **({"ContinuationToken": token} if token else {})}
            resp = self.client.list_objects_v2(**kw)
            out += [o["Key"] for o in resp.get("Contents", [])]
            if not resp.get("IsTruncated"):
                return sorted(out)
            token = resp.get("NextContinuationToken")


def make_object_store(settings: Settings):
    if settings.storage_backend == "s3" and settings.s3_bucket:
        return S3ObjectStore(settings)
    return LocalObjectStore(Path(settings.data_dir) / "objects")
