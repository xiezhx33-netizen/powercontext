# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned, digest-checked context packs and append-only deltas."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    """Raised when a context pack or delta cannot be trusted or reconstructed."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _b64(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(values: object) -> tuple[bytes, ...]:
    if not isinstance(values, list):
        raise ContextPackError("items must be a list")
    decoded = []
    for value in values:
        if not isinstance(value, str):
            raise ContextPackError("items must be base64 strings")
        try:
            decoded.append(base64.b64decode(value.encode("ascii"), validate=True))
        except Exception as error:
            raise ContextPackError("invalid base64 item") from error
    return tuple(decoded)


def _finalize(payload: dict[str, object]) -> bytes:
    payload = dict(payload)
    payload["byte_count"] = 0
    for _ in range(8):
        data = _canonical(payload)
        if payload["byte_count"] == len(data):
            return data
        payload["byte_count"] = len(data)
    raise ContextPackError("byte_count did not stabilize")


@dataclass(frozen=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        unsigned = {"schema": PACK_SCHEMA, "items": _b64(items)}
        digest = "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()
        pack = cls(items=tuple(items), digest=digest, byte_count=0)
        object.__setattr__(pack, "byte_count", len(pack.to_bytes()))
        return pack

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")
        data = content.encode("utf-8")
        items: list[bytes] = []
        start = 0
        while start < len(data):
            end = min(start + chunk_bytes, len(data))
            while end > start and end < len(data) and 0x80 <= data[end] <= 0xBF:
                end -= 1
            if end == start:
                raise ContextPackError("chunk_bytes cannot split a UTF-8 character")
            items.append(data[start:end])
            start = end
        return cls.build(tuple(items))

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        try:
            payload = json.loads(data)
        except Exception as error:
            raise ContextPackError("context pack must be valid JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"schema", "items", "digest", "byte_count"}:
            raise ContextPackError("context pack has an invalid schema")
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("unsupported context pack schema")
        items = _decode_items(payload["items"])
        unsigned = {"schema": PACK_SCHEMA, "items": payload["items"]}
        digest = "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()
        if payload["digest"] != digest:
            raise ContextPackError("context pack digest mismatch")
        if not isinstance(payload["byte_count"], int) or payload["byte_count"] != len(data):
            raise ContextPackError("context pack byte_count mismatch")
        return cls(items=items, digest=digest, byte_count=len(data))

    def to_bytes(self) -> bytes:
        payload = {"schema": PACK_SCHEMA, "items": _b64(self.items), "digest": self.digest}
        return _finalize(payload)


@dataclass(frozen=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]
    byte_count: int = 0

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        while prefix < len(base.items) and prefix < len(target.items) and base.items[prefix] == target.items[prefix]:
            prefix += 1
        delta = cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
            byte_count=0,
        )
        object.__setattr__(delta, "byte_count", len(delta.to_bytes()))
        return delta

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        try:
            payload = json.loads(data)
        except Exception as error:
            raise ContextPackError("context pack delta must be valid JSON") from error
        expected = {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ContextPackError("context pack delta has an invalid schema")
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("unsupported context pack delta schema")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not isinstance(target_digest, str):
            raise ContextPackError("context pack delta digests must be strings")
        prefix = payload["prefix_items"]
        if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        appended = _decode_items(payload["appended"])
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix,
            appended=appended,
            byte_count=len(data),
        )

    def to_bytes(self) -> bytes:
        payload = {
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _b64(self.appended),
        }
        return _canonical(payload)


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    if base.digest != delta.base_digest:
        raise ContextPackError("context pack delta base mismatch")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("context pack delta prefix exceeds base")
    rebuilt = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("context pack delta target mismatch")
    return rebuilt


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]