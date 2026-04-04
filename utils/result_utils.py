import logging

from schema import EntityRow, SourceCitation, SourceRecord, StructuredEntityTable, TableCell, TableColumn
from utils.text_utils import compact_text, normalize_key, slugify

logger = logging.getLogger(__name__)


def normalize_result(
    result: StructuredEntityTable,
    query: str,
    source_registry: dict[str, dict[str, str | None]],
    require_complete: bool = True,
    required_column_keys: set[str] | None = None,
) -> StructuredEntityTable:
    normalized_columns: list[TableColumn] = []
    seen_column_keys: set[str] = set()

    for column in result.columns:
        key = normalize_key(column.key or column.label)
        if not key or key in seen_column_keys:
            continue
        normalized_columns.append(
            TableColumn(
                key=key,
                label=compact_text(column.label) or key.replace("_", " ").title(),
                description=compact_text(column.description) or "",
            )
        )
        seen_column_keys.add(key)

    if "name" not in seen_column_keys:
        normalized_columns.insert(
            0,
            TableColumn(
                key="name",
                label="Name",
                description="Name of the discovered entity",
            ),
        )
        seen_column_keys.add("name")

    normalized_rows: list[EntityRow] = []
    used_source_ids: set[str] = set()
    dropped_incomplete_rows = 0

    for row in result.rows:
        raw_cells = {normalize_key(key): cell for key, cell in (row.cells or {}).items()}
        cleaned_cells: dict[str, TableCell] = {}
        row_source_ids: set[str] = set()

        for column in normalized_columns:
            cell = raw_cells.get(column.key)
            if not cell or not compact_text(cell.value):
                continue

            citations: list[SourceCitation] = []
            for citation in cell.citations or []:
                source_id = compact_text(citation.source_id)
                source_meta = source_registry.get(source_id or "", {})
                source_title = compact_text(citation.source_title) or compact_text(
                    source_meta.get("title")
                )
                source_url = compact_text(citation.source_url) or compact_text(
                    source_meta.get("url")
                )
                quote = compact_text(citation.quote)

                if not source_id or not source_title or not source_url or not quote:
                    continue

                citations.append(
                    SourceCitation(
                        source_id=source_id,
                        source_title=source_title,
                        source_url=source_url,
                        quote=quote,
                    )
                )
                row_source_ids.add(source_id)

            if not citations:
                continue

            cleaned_cells[column.key] = TableCell(
                value=compact_text(cell.value),
                citations=citations,
            )

        name_cell = cleaned_cells.get("name")
        if not name_cell or not name_cell.value or not name_cell.citations:
            continue

        required_keys = required_column_keys or {
            column.key for column in normalized_columns
        }
        if require_complete and any(
            key not in cleaned_cells
            or not cleaned_cells[key].value
            or not cleaned_cells[key].citations
            for key in required_keys
        ):
            dropped_incomplete_rows += 1
            continue

        normalized_rows.append(
            EntityRow(
                entity_id=slugify(name_cell.value or row.entity_id),
                cells=cleaned_cells,
            )
        )
        used_source_ids.update(row_source_ids)

    normalized_sources = [
        SourceRecord(
            source_id=source_id,
            title=compact_text(source_registry[source_id].get("title")) or source_id,
            url=compact_text(source_registry[source_id].get("url")) or "",
            snippet=compact_text(source_registry[source_id].get("snippet")),
        )
        for source_id in sorted(used_source_ids)
        if source_id in source_registry and compact_text(source_registry[source_id].get("url"))
    ]

    if require_complete and dropped_incomplete_rows:
        logger.info(
            "dropped %d incomplete row(s) from final result because some column values were missing",
            dropped_incomplete_rows,
        )

    return StructuredEntityTable(
        query=query,
        title=compact_text(result.title) or f"Discovered entities for {query}",
        columns=normalized_columns,
        rows=normalized_rows,
        sources=normalized_sources,
    )


def collect_sources_from_rows(
    rows: list[EntityRow], source_registry: dict[str, dict[str, str | None]]
) -> list[SourceRecord]:
    used_source_ids: set[str] = set()
    for row in rows:
        for cell in row.cells.values():
            for citation in cell.citations:
                if citation.source_id:
                    used_source_ids.add(citation.source_id)

    return [
        SourceRecord(
            source_id=source_id,
            title=compact_text(source_registry[source_id].get("title")) or source_id,
            url=compact_text(source_registry[source_id].get("url")) or "",
            snippet=compact_text(source_registry[source_id].get("snippet")),
        )
        for source_id in sorted(used_source_ids)
        if source_id in source_registry
        and compact_text(source_registry[source_id].get("url"))
    ]
