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

"""Parse Pi v3 session branches into immutable, row-level evidence Sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    Connector,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

_SESSION_TYPE = "session"
_PI_VERSION = 3
_CONTROL_TYPES = {
    "model_change",
    "thinking_level_change",
    "session",
    "label",
    "branch",
}


class PiSessionInput(BaseModel):
    """One raw JSONL archive and the branch leaf to materialize."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One immutable visible event on the selected branch."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    """A captured Pi event with its original record metadata."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _message_body(message: dict[str, Any]) -> str:
    role = message.get("role")
    if role == "bashExecution":
        command = message.get("command") or ""
        output = message.get("output") or ""
        return f"$ {command}\n{output}"
    if role == "toolResult":
        tool_name = message.get("toolName") or ""
        call_id = message.get("toolCallId") or ""
        text = _text_from_content(message.get("content"))
        return f"[tool_result {tool_name} {call_id}] {text}"
    if role == "assistant":
        parts: list[str] = []
        text = _text_from_content(message.get("content"))
        if text:
            parts.append(text)
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    name = block.get("name") or ""
                    call_id = block.get("id") or ""
                    args = block.get("arguments") or {}
                    parts.append(f"[tool_call {name} {call_id}] {_compact_json(args)}")
        return "\n".join(parts)
    return _text_from_content(message.get("content"))


def _visible_event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    event_type = record.get("type")
    record_id = record.get("id")
    parent_id = record.get("parentId")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Pi record id must be non-empty")
    role: str | None = None
    body = ""
    if event_type == "message" and isinstance(record.get("message"), dict):
        message = record["message"]
        role = message.get("role")
        body = _message_body(message)
    elif event_type in {"compaction", "branch_summary"}:
        role = event_type
        body = record.get("summary") if isinstance(record.get("summary"), str) else ""
    elif event_type == "custom_message":
        role = "custom"
        body = _text_from_content(record.get("content"))
    elif event_type in _CONTROL_TYPES:
        return None
    else:
        return None
    if not isinstance(role, str) or not role or not body:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=record_id,
        parent_id=parent_id if isinstance(parent_id, str) else None,
        event_type=str(event_type),
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse the selected leaf's complete root-to-leaf visible branch."""

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session must be valid UTF-8") from error
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Pi session contains invalid JSONL") from error
        if not isinstance(row, dict):
            raise ValueError("Pi session rows must be JSON objects")
        rows.append(row)
    if not rows:
        raise ValueError("Pi session is empty")
    header = rows[0]
    if (
        header.get("type") != _SESSION_TYPE
        or header.get("version") != _PI_VERSION
        or not isinstance(header.get("id"), str)
        or not header["id"].strip()
    ):
        raise ValueError("Pi session header must be type=session, version=3 with a non-empty id")
    session_id = header["id"]
    records: dict[str, dict[str, Any]] = {}
    last_id: str | None = None
    for row in rows[1:]:
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Pi record id must be a non-empty string")
        if record_id in records:
            raise ValueError("Pi record ids must be unique")
        parent_id = row.get("parentId")
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in records):
            raise ValueError("Pi parent must reference an earlier record")
        records[record_id] = row
        last_id = record_id
    if not records:
        raise ValueError("Pi session has no records")
    leaf = value.leaf_id if value.leaf_id is not None else last_id
    if leaf not in records:
        raise ValueError("Pi leaf does not exist")
    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = records[leaf]
    while True:
        current_id = current["id"]
        if current_id in seen:
            raise ValueError("Pi session contains a cycle")
        seen.add(current_id)
        path.append(current)
        parent_id = current.get("parentId")
        if parent_id is None:
            break
        current = records[parent_id]
    path.reverse()
    events: list[PiEventInput] = []
    for record in path:
        event = _visible_event(session_id, record)
        if event is not None:
            events.append(event)
    return tuple(events)


def pi_event_source_id(session_id: str, event_id: str) -> str:
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("Pi event identity exceeds the Source id limit")
    return identity


class _PiEventAdapter:
    input_class = PiEventInput
    source_class = PiEventSource
    name = "pi-event"

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            description=f"Pi event {value.event_id!r} from {value.session_id!r}",
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class _PiEventTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> dict[str, object]:
        return {
            "source_type": "pi-event",
            "source_id": source.source_id,
            "content": source.content,
            "metadata": {
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        }


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    adapter=_PiEventAdapter(),
    version="1",
    projections=(_PiEventTextEvidenceProjection(),),
)


def _digest(event: PiEventInput) -> str:
    payload = json.dumps(
        {"content": event.content, "record": event.record},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PiSessionConnector:
    """Import only visible events not already covered by a durable checkpoint."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = self._path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=raw, leaf_id=self._leaf_id))
        previous = session.checkpoint if isinstance(session.checkpoint, dict) else {}
        previous_events = previous.get("events") if isinstance(previous.get("events"), dict) else {}
        current: dict[str, str] = {}
        for event in events:
            current[event.event_id] = _digest(event)
        for event_id, digest in previous_events.items():
            if event_id in current and current[event_id] != digest:
                raise ValueError("a previously imported Pi event changed")
        for event in events:
            if event.event_id in previous_events:
                continue
            await session.submit(event.event_id, "pi-event", event)
        checkpoint = {"session_id": events[0].session_id if events else None, "events": current}
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]