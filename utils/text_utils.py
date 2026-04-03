import re

from schema import StructuredEntityTable
from utils.config import MAX_CONTENT_CHARS


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "entity"


def compact_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split()).strip()
    return cleaned or None


def truncate_content(value: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    trimmed = value[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{trimmed}\n\n[...truncated]"


def escape_markdown_cell(value: str) -> str:
    compact = " ".join(value.split())
    if len(compact) > 100:
        compact = f"{compact[:97].rstrip()}..."
    return compact.replace("|", "\\|")


def render_table_markdown(result: StructuredEntityTable) -> str:
    if not result.rows:
        return "_No entities extracted._"

    headers = [column.label for column in result.columns]
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---'] * len(headers))} |",
    ]

    for row in result.rows:
        normalized_cells = {
            normalize_key(key): cell for key, cell in (row.cells or {}).items()
        }
        values = []
        for column in result.columns:
            cell = normalized_cells.get(column.key)
            values.append(escape_markdown_cell(cell.value) if cell and cell.value else "")
        lines.append(f"| {' | '.join(values)} |")

    return "\n".join(lines)


def render_markdown_document(result: StructuredEntityTable) -> str:
    lines = [
        f"# {result.title}",
        "",
        f"Query: `{result.query}`",
        "",
        "## Entity Table",
        render_table_markdown(result),
        "",
        "## Sources",
    ]

    for source in result.sources:
        lines.append(f"- `{source.source_id}`: [{source.title}]({source.url})")

    return "\n".join(lines)
