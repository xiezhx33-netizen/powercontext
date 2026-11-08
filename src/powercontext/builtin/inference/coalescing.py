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

"""Coalesce concurrent embedding requests, deduplicate texts, and cache vectors."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Sequence

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage


class CoalescingEmbeddingModel:
    """A bounded-batch embedding wrapper with in-flight sharing and an LRU cache."""

    def __init__(
        self,
        base,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 0,
        cache_size: int = 1024,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must not be negative")
        self._base = base
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._in_flight: dict[str, asyncio.Future] = {}
        self._pending: set[str] = set()
        self._flush_task: asyncio.Task | None = None
        self._call_for_text: dict[str, int] = {}
        self._call_usage: dict[int, InferenceUsage] = {}
        self._next_call_id = 0

    @property
    def profile(self):
        return self._base.profile

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        requested = tuple(texts)
        if not requested:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        unique = tuple(dict.fromkeys(requested))
        futures: dict[str, asyncio.Future] = {}
        missing: list[str] = []
        for text in unique:
            cached = self._cache_get(text)
            if cached is not None:
                continue
            missing.append(text)
            future = self._in_flight.get(text)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._in_flight[text] = future
                self._pending.add(text)
            futures[text] = future
        if self._pending:
            self._schedule_flush()

        values: dict[str, tuple[float, ...]] = {}
        for text in unique:
            cached = self._cache_get(text)
            if cached is not None:
                values[text] = cached
            else:
                values[text] = await asyncio.shield(futures[text])

        call_ids = {self._call_for_text[text] for text in missing if text in self._call_for_text}
        requests = len(call_ids)
        input_tokens = None
        output_tokens = None
        if call_ids:
            input_tokens = sum(
                self._call_usage[call_id].input_tokens or 0 for call_id in call_ids
            )
            output_tokens = sum(
                self._call_usage[call_id].output_tokens or 0 for call_id in call_ids
            )
        return EmbeddingResult(
            vectors=tuple(values[text] for text in requested),
            usage=InferenceUsage(
                requests=requests,
                input_tokens=input_tokens if requests else None,
                output_tokens=output_tokens if requests else None,
            ),
        )

    def _cache_get(self, text: str) -> tuple[float, ...] | None:
        if self._cache_size == 0:
            return None
        value = self._cache.get(text)
        if value is not None:
            self._cache.move_to_end(text)
        return value

    def _cache_put(self, text: str, vector: tuple[float, ...]) -> None:
        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _schedule_flush(self) -> None:
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush())

    async def _flush(self) -> None:
        if self._coalesce_seconds:
            await asyncio.sleep(self._coalesce_seconds)
        else:
            await asyncio.sleep(0)
        while self._pending:
            chunk = tuple(self._pending)[: self._max_batch_size]
            for text in chunk:
                self._pending.discard(text)
            await self._dispatch(chunk)

    async def _dispatch(self, chunk: tuple[str, ...]) -> None:
        call_id = self._next_call_id
        self._next_call_id += 1
        try:
            result = await self._base.embed(chunk)
            if len(result.vectors) != len(chunk):
                raise ValueError("embedding provider returned the wrong number of vectors")
            for text, vector in zip(chunk, result.vectors, strict=True):
                tuple_vector = tuple(float(value) for value in vector)
                self._call_for_text[text] = call_id
                self._cache_put(text, tuple_vector)
                future = self._in_flight.pop(text, None)
                if future is not None and not future.done():
                    future.set_result(tuple_vector)
            self._call_usage[call_id] = result.usage
        except Exception as error:
            for text in chunk:
                future = self._in_flight.pop(text, None)
                if future is not None and not future.done():
                    future.set_exception(error)


__all__ = ["CoalescingEmbeddingModel"]