from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.schemas import ThreatModelingReport


class ReportGenerator:
    """Generate PDF report from a ThreatModelingReport using ReportLab."""

    def __init__(self) -> None:
        self.styles = getSampleStyleSheet()

        # Avoid clashes / duplicates (e.g., hot reload re-instantiations)
        if "TM_Heading1" not in self.styles:
            self.styles.add(ParagraphStyle(name="TM_Heading1", fontSize=18, leading=22))
        if "TM_Heading2" not in self.styles:
            self.styles.add(ParagraphStyle(name="TM_Heading2", fontSize=14, leading=18))
        if "TM_Normal" not in self.styles:
            self.styles.add(ParagraphStyle(name="TM_Normal", fontSize=10, leading=12))

    def generate(self, report: ThreatModelingReport, output_path: str) -> str:
        """
        Produce PDF at the given path and return the full filepath.
        Creates directories as needed and ensures the path ends in .pdf.
        """

        def _sanitize_filename(p: Path) -> Path:
            name = p.name
            safe = "".join(c for c in name if c.isalnum() or c in " ._-")
            return p.with_name(safe)

        out_path = _sanitize_filename(Path(output_path))
        if out_path.suffix.lower() != ".pdf":
            out_path = out_path.with_suffix(".pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(str(out_path), pagesize=LETTER)
        elements = []

        # Cover
        elements.append(Paragraph(report.meta.title, self.styles["TM_Heading1"]))
        if report.meta.system_name:
            elements.append(Paragraph(report.meta.system_name, self.styles["TM_Heading2"]))
        elements.append(
            Paragraph(f"Generated: {report.meta.generated_at_iso}", self.styles["TM_Normal"])
        )
        elements.append(PageBreak())

        # Detecções
        elements.append(Paragraph("Detecções", self.styles["TM_Heading2"]))
        det_table = [["ID", "Componente", "Confiança"]]
        for d in report.diagram.detections:
            det_table.append([d.id, d.label.value, f"{d.confidence:.2f}"])

        table = Table(det_table, hAlign="LEFT", colWidths=[150, 150, 80])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]
            )
        )
        elements.append(table)
        elements.append(PageBreak())

        # Ameaças STRIDE (tabela resumo)
        elements.append(Paragraph("Ameaças STRIDE", self.styles["TM_Heading2"]))
        thr_table = [["Componente", "Categoria", "Severidade", "Título", "Conf"]]
        for t in report.stride.threats:
            title = t.title if len(t.title) <= 140 else t.title[:137] + "..."
            thr_table.append(
                [
                    t.component_label.value,
                    t.category.value,
                    t.severity.value,
                    title,
                    f"{t.confidence:.2f}",
                ]
            )

        table = Table(thr_table, hAlign="LEFT", repeatRows=1, colWidths=[120, 100, 80, 150, 50])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]
            )
        )
        elements.append(table)

        # Threat details
        for t in report.stride.threats:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(t.title, self.styles["TM_Heading2"]))
            elements.append(
                Paragraph(f"Componente: {t.component_label.value}", self.styles["TM_Normal"])
            )
            elements.append(Paragraph(f"Categoria: {t.category.value}", self.styles["TM_Normal"]))
            elements.append(Paragraph(f"Severidade: {t.severity.value}", self.styles["TM_Normal"]))

            elements.append(Paragraph("Descrição:", self.styles["TM_Normal"]))
            elements.append(Paragraph(t.description, self.styles["TM_Normal"]))

            elements.append(Paragraph("Impacto:", self.styles["TM_Normal"]))
            elements.append(Paragraph(t.impact, self.styles["TM_Normal"]))

            if t.mitigations:
                elements.append(Paragraph("Mitigações:", self.styles["TM_Normal"]))
                items = []
                for m in t.mitigations:
                    txt = m if len(m) <= 160 else m[:157] + "..."
                    items.append(ListItem(Paragraph(txt, self.styles["TM_Normal"])))
                elements.append(ListFlowable(items, bulletType="bullet"))

        elements.append(PageBreak())

        # Enriquecimento (RAG)
        if report.enrichment:
            elements.append(Paragraph("Enriquecimento (RAG)", self.styles["TM_Heading2"]))
            for e in report.enrichment.enrichments:
                elements.append(Paragraph(f"Threat ID: {e.threat_id}", self.styles["TM_Normal"]))
                for hit in e.kb_hits:
                    snippet = (hit.snippet or "")
                    if len(snippet) > 120:
                        snippet = snippet[:117] + "..."
                    elements.append(
                        Paragraph(
                            f"{hit.source}/{hit.key} ({hit.score:.2f}): {snippet}",
                            self.styles["TM_Normal"],
                        )
                    )

        doc.build(elements)
        return str(out_path)