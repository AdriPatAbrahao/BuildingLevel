from results.resultsext import (
    extract_material_breakdown,
    extract_material_summary,
)


def _report_html(headers, totals):
    header_cells = "".join(f"<td>{value}</td>" for value in headers)
    total_cells = "".join(f"<td>{value}</td>" for value in totals)
    return f"""
    <html><body>
      <table><tr><td>Outra tabela</td></tr></table>
      <table>
        <tr><td colspan="{len(headers)}">Resumo de materiais</td></tr>
        <tr>{header_cells}</tr>
        <tr>{total_cells}</tr>
      </table>
    </body></html>
    """


def test_extracts_values_by_header_instead_of_fixed_position(tmp_path):
    report = tmp_path / "RESDES.HTM"
    report.write_text(
        _report_html(
            ["Elemento", "Concreto", "Campo novo", "Aço", "Forma"],
            ["Totais", "18.51", "999", "1438", "186.75"],
        ),
        encoding="utf-8",
    )

    assert extract_material_summary(report) == ("1438", "18.51")


def test_reads_cp1252_report_and_normalizes_accented_header(tmp_path):
    report = tmp_path / "RESDES_CP1252.HTM"
    html = _report_html(
        ["Elemento", "Aço", "Concreto"],
        ["Totais", "1500", "20,25"],
    )
    report.write_bytes(html.encode("cp1252"))

    assert extract_material_summary(report) == ("1500", "20,25")


def test_missing_report_returns_unpackable_empty_summary(tmp_path):
    assert extract_material_summary(tmp_path / "missing.htm") == (None, None)


def test_incompatible_table_returns_empty_summary(tmp_path):
    report = tmp_path / "invalid.htm"
    report.write_text(
        "<table><tr><td>Resumo de materiais</td></tr>"
        "<tr><td>Totais</td><td>1438</td></tr></table>",
        encoding="utf-8",
    )

    assert extract_material_summary(report) == (None, None)


def test_extracts_element_material_breakdown(tmp_path):
    report = tmp_path / "RESDES.HTM"
    report.write_text(
        _report_html(
            ["Elemento", "Aço", "Concreto", "Forma", "fck"],
            ["Totais", "238", "11.18", "114.36", "-"],
        ).replace(
            "<tr><td>Totais</td><td>238</td><td>11.18</td>"
            "<td>114.36</td><td>-</td></tr>",
            "<tr><td>Pilares</td><td>238</td><td>1.86</td>"
            "<td>29.40</td><td>25</td></tr>"
            "<tr><td>Totais</td><td>238</td><td>11.18</td>"
            "<td>114.36</td><td>-</td></tr>",
        ),
        encoding="utf-8",
    )

    result = extract_material_breakdown(report)

    assert result["columns"] == {
        "steel_kgf": "238",
        "concrete_m3": "1.86",
        "formwork_m2": "29.40",
        "fck_mpa": "25",
    }
    assert result["total"] == {
        "steel_kgf": "238",
        "concrete_m3": "11.18",
        "formwork_m2": "114.36",
    }
