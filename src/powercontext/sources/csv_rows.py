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

"""Row-level CSV Sources and their standard text evidence projection."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class CSVDocumentInput(BaseModel):
    """A complete, offline CSV document to split into row Sources."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One parsed CSV row, preserving header order and raw values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A materialized row Source referenced by its stable document/key identity."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def _validate_document_id(document_id: str) -> None:
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be a non-empty string without surrounding whitespace")


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the delimiter-safe, bounded identity of one CSV row."""

    _validate_document_id(document_id)
    if not isinstance(primary_key_value, str):
        raise ValueError("primary_key_value must be a string")
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"csv row source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return source_id


def _parse_records(text: str) -> list[list[str]]:
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        return list(reader)
    except csv.Error as error:
        raise ValueError(f"malformed CSV: {error}") from error


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate every row before returning any partial result."""

    _validate_document_id(value.document_id)
    if not isinstance(value.primary_key, str) or not value.primary_key:
        raise ValueError("primary_key must be a non-empty string")

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error

    records = _parse_records(text)
    if not records:
        raise ValueError("CSV document must contain a header")
    header = records[0]
    if not header:
        raise ValueError("CSV header must not be empty")
    if any(not column or not column.strip() for column in header):
        raise ValueError("CSV header names must be non-empty and non-blank")
    if len(set(header)) != len(header):
        raise ValueError("CSV header names must be unique")
    if value.primary_key not in header:
        raise ValueError(f"primary key column {value.primary_key!r} is not present in the header")
    key_index = header.index(value.primary_key)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in records[1:]:
        # csv.reader yields [] for a completely empty physical record.  Such
        # records are ignored, while whitespace in a real field is preserved.
        if not record:
            continue
        if len(record) != len(header):
            raise ValueError("CSV data row width does not match the header")
        primary_key_value = record[key_index]
        if not primary_key_value or not primary_key_value.strip():
            raise ValueError("CSV primary key values must be non-empty and non-blank")
        if primary_key_value in seen_keys:
            raise ValueError(f"duplicate CSV primary key value {primary_key_value!r}")
        seen_keys.add(primary_key_value)
        # Validate the identity length now so a bad document cannot produce a
        # partially accepted set of rows.
        csv_row_source_id(value.document_id, primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=tuple(header),
                values=tuple(record),
            )
        )
    return tuple(rows)


class _CSVRowAdapter:
    input_class = CSVRowInput
    source_class = CSVRowSource
    name = "csv-row"

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            description=f"CSV row {value.primary_key_value!r} from {value.document_id!r}",
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class _CSVRowTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> dict[str, object]:
        row = dict(zip(source.columns, source.values, strict=True))
        return {
            "source_type": "csv-row",
            "source_id": source.source_id,
            "content": json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            "metadata": {
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        }


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    adapter=_CSVRowAdapter(),
    version="1",
    projections=(_CSVRowTextEvidenceProjection(),),
)


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]