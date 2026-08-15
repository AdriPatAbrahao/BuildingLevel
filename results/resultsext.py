"""Extraction of material totals from the TQS ``RESDES.HTM`` report."""

from pathlib import Path
from typing import Optional, Tuple
import unicodedata

from bs4 import BeautifulSoup


MaterialSummary = Tuple[Optional[str], Optional[str]]


def _normalize_label(value: str) -> str:
    """Normalize a report label for accent- and case-insensitive matching."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(without_accents.casefold().split())


def _read_report(path: Path) -> Optional[str]:
    """Read a TQS HTML report using the encodings observed in supported files."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def extract_material_summary(file_path) -> MaterialSummary:
    """Extract total steel and concrete values from a TQS material table.

    Columns are located by the normalized headers ``Aço`` and ``Concreto``;
    their numerical positions may therefore change between report versions.
    The result values remain strings because numeric conversion and locale
    handling belong to the calling analysis workflow.

    Parameters
    ----------
    file_path:
        Path to a TQS ``RESDES.HTM`` report.

    Returns
    -------
    tuple[Optional[str], Optional[str]]
        ``(steel_value, concrete_value)``. On a missing file or incompatible
        table schema, returns ``(None, None)`` so callers can safely unpack it.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {path}")
        return None, None

    html_content = _read_report(path)
    if html_content is None:
        print("Failed to read the file with the tested encodings.")
        return None, None

    soup = BeautifulSoup(html_content, "lxml")

    target_table = None
    for table in soup.find_all("table"):
        labels = [
            _normalize_label(cell.get_text(" ", strip=True))
            for cell in table.find_all(["td", "th"])
        ]
        if "resumo de materiais" in labels:
            target_table = table
            break

    if target_table is None:
        print("Table 'Resumo de materiais' not found.")
        return None, None

    steel_index = None
    concrete_index = None
    totals_cells = None

    for row in target_table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        labels = [
            _normalize_label(cell.get_text(" ", strip=True))
            for cell in cells
        ]
        if "aco" in labels and "concreto" in labels:
            steel_index = labels.index("aco")
            concrete_index = labels.index("concreto")
        if labels and labels[0].startswith("totais"):
            totals_cells = cells

    if steel_index is None or concrete_index is None:
        print("Headers 'Aço' and 'Concreto' not found in material summary.")
        return None, None
    if totals_cells is None:
        print("Row 'Totais' not found in the table.")
        return None, None

    required_index = max(steel_index, concrete_index)
    if len(totals_cells) <= required_index:
        print("Error: The totals row does not match the material headers.")
        return None, None

    steel_value = totals_cells[steel_index].get_text(" ", strip=True)
    concrete_value = totals_cells[concrete_index].get_text(" ", strip=True)
    if steel_value in {"", "-"} or concrete_value in {"", "-"}:
        print("Error: Steel or concrete total is empty in the material summary.")
        return None, None

    return steel_value, concrete_value

