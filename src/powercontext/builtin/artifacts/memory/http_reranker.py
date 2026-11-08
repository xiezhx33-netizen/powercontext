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

"""HTTP cross-encoder reranking with bounded batches and global ordering."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score Memory candidates with an HTTP service, then select globally."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not endpoint:
            raise ValueError("endpoint must be a non-empty string")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        if not candidates:
            raise ValueError("rerank requires at least one candidate")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = tuple(start + offset + 1 for offset in range(len(batch)))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text}
                    for rank, candidate in zip(ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            requests += 1
            response.raise_for_status()
            try:
                body = response.json()
            except ValueError as error:
                raise ValueError("reranker response must be valid JSON") from error
            scores.update(_parse_scores(body, ranks))

        selected = tuple(
            rank
            for _score, rank in sorted(
                ((-score, rank) for rank, score in scores.items()),
            )[:limit]
        )
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=requests),
        )


def _parse_scores(body: Any, expected_ranks: tuple[int, ...]) -> dict[int, float]:
    if not isinstance(body, dict) or set(body) != {"scores"}:
        raise ValueError("reranker response must contain exactly the scores field")
    raw_scores = body["scores"]
    if not isinstance(raw_scores, list):
        raise ValueError("reranker scores must be a list")
    scores: dict[int, float] = {}
    for item in raw_scores:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("each reranker score must contain exactly rank and score")
        rank = item["rank"]
        score = item["score"]
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("reranker rank must be an integer")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError("reranker score must be a finite number")
        if rank not in expected_ranks or rank in scores:
            raise ValueError("reranker scores must cover each batch rank exactly once")
        scores[rank] = float(score)
    if set(scores) != set(expected_ranks):
        raise ValueError("reranker scores must cover each batch rank exactly once")
    return scores


__all__ = ["HTTPMemoryReranker"]