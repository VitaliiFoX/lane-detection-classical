"""Збирає звіт у форматі звітів до лаб: титулка з шаблону + docs/report/zvit.md → .docx і .pdf.

Потрібні pandoc, LibreOffice (soffice) і python-docx. Рисунки беруться з results/,
тож спершу `python main.py run --pipeline all` і `python main.py report`.
Запуск: `python tools/build_report.py` → results/zvit.docx, results/zvit.pdf.
"""

import copy
import re
import subprocess
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "report" / "zvit.md"
TEMPLATE = ROOT / "docs" / "report" / "shablon.docx"
OUT = ROOT / "results" / "zvit.docx"


def markdown_body():
    """Текст звіту від анотації: назва й примітка для читачів репозиторію — на титулці."""
    text = SOURCE.read_text(encoding="utf-8")
    return text[text.index("## Анотація"):]


def run_pandoc(target):
    # Розділи «##» стають Heading 1, як у звітах до лаб; стилі — з шаблону.
    subprocess.run(["pandoc", "-f", "markdown", "-o", str(target), "--reference-doc", str(TEMPLATE),
                    "--shift-heading-level-by=-1", "--resource-path", str(ROOT)],
                   input=markdown_body(), text=True, check=True)


def add_title_page(document):
    """Копіює абзаци титулки з шаблону на початок документа."""
    title = docx.Document(str(TEMPLATE)).element.body
    first = document.element.body[0]
    for element in title:
        if element.tag != qn("w:sectPr"):
            first.addprevious(copy.deepcopy(element))


def style_table(table):
    """Сірі рамки D9D9D9 і заголовок E8EBEF, як у звітах до лаб."""
    for r, row in enumerate(table.rows):
        for cell in row.cells:
            props = cell._tc.get_or_add_tcPr()  # pylint: disable=protected-access
            borders = OxmlElement("w:tcBorders")
            for side in ("top", "left", "bottom", "right"):
                edge = OxmlElement(f"w:{side}")
                edge.set(qn("w:val"), "single")
                edge.set(qn("w:sz"), "4")
                edge.set(qn("w:color"), "D9D9D9")
                borders.append(edge)
            props.append(borders)
            if r == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:val"), "clear")
                shade.set(qn("w:fill"), "E8EBEF")
                props.append(shade)


def use_template_styles(document):
    """Стилі pandoc, яких немає в шаблоні, замінюємо: без цього LibreOffice розсипає таблиці."""
    known = {style.style_id for style in document.styles}
    mapping = {"SourceCode": "Listing", "ImageCaption": "Caption"}
    body = document.element.body
    for tag in ("w:pStyle", "w:rStyle"):
        for element in list(body.iter(qn(tag))):
            value = element.get(qn("w:val"))
            if value in mapping:
                element.set(qn("w:val"), mapping[value])
            elif value not in known:
                element.getparent().remove(element)


def polish(document):
    """Оформлення як у звітах до лаб: розділи з нової сторінки, підписи, таблиці."""
    use_template_styles(document)
    first_heading = True
    for paragraph in document.paragraphs:
        style = paragraph.style.name
        if style == "Heading 1":
            # Кожен розділ з нової сторінки; титулка вже закінчується розривом.
            paragraph.paragraph_format.page_break_before = not first_heading
            first_heading = False
        elif style == "Image Caption" or re.match(r"Рисунок \d+ — ", paragraph.text):
            paragraph.style = document.styles["Caption"]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif paragraph._p.xpath(".//w:drawing"):  # pylint: disable=protected-access
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for table in document.tables:
        style_table(table)
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(2)
                    paragraph.paragraph_format.line_spacing = 1.0
                    for run in paragraph.runs:
                        run.font.size = Pt(10.5)


def main():
    """pandoc → титулка й оформлення → LibreOffice робить PDF."""
    OUT.parent.mkdir(exist_ok=True)
    run_pandoc(OUT)
    document = docx.Document(str(OUT))
    add_title_page(document)
    polish(document)
    document.save(str(OUT))
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT.parent),
                    str(OUT)], check=True, capture_output=True)
    print("Звіт:", OUT, "і", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
