#!/usr/bin/env python3
"""Render the v6.4 negative-sample markdown into a one-sample-per-page PDF.

Each page shows the sample id and type, the NMR text, and the correct /
negative SMILES together with RDKit structure drawings.
"""
from __future__ import annotations

import argparse
import io
import re
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = ROOT / "docs" / "v6_4_negative_samples_visualization.md"
DEFAULT_PDF = ROOT / "docs" / "v6_4_negative_samples_visualization.pdf"

# ReportLab cannot embed the Noto CJK CFF collections, so Chinese uses WenQuanYi.
FONT_REG = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
FONT_BOLD = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
FONT_MONO = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"

PAGE_W, PAGE_H = A4
MARGIN = 28

CORRECT_COLOR = HexColor("#047857")
NEGATIVE_COLOR = HexColor("#B91C1C")
INK = HexColor("#111827")
MUTED = HexColor("#4B5563")
LINE = HexColor("#E5E7EB")
NMR_BG = HexColor("#F8FAFC")
PANEL_BG = HexColor("#FFFFFF")

GROUP_STYLE = {
    "random": ("随机负样本", HexColor("#1D4ED8")),
    "same_formula": ("同分子式负样本", HexColor("#047857")),
    "similar_structure": ("相似结构负样本", HexColor("#B45309")),
    "perturbation": ("微扰负样本", HexColor("#7E22CE")),
}


@dataclass
class Sample:
    sample_id: str
    group: str
    sample_type: str
    correct_smiles: str
    negative_smiles: str
    quality: str
    nmr_text: str


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("NotoSans", FONT_REG, subfontIndex=0))
    pdfmetrics.registerFont(TTFont("NotoSans-Bold", FONT_BOLD, subfontIndex=0))
    pdfmetrics.registerFont(TTFont("NotoMono", FONT_MONO))


def parse_markdown(path: Path) -> tuple[str, list[str], list[Sample]]:
    text = path.read_text(encoding="utf-8")
    title = "v6.4 负样本可视化"
    preamble: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            break
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            preamble.append(stripped[2:].strip())
    title = title.replace("实际", "").replace("高质量", "")
    preamble = [
        item.replace("实际训练源", "训练源")
        for item in preamble
        if "训练源" in item
    ]

    samples: list[Sample] = []
    chunks = re.split(r"(?=^### )", text, flags=re.M)
    for chunk in chunks:
        if not chunk.startswith("### "):
            continue
        lines = chunk.splitlines()
        sample_id = lines[0][4:].strip()
        group = sample_id.rsplit("-", 1)[0]
        fields: dict[str, str] = {}
        nmr_at: int | None = None
        for i, line in enumerate(lines[1:], start=1):
            if not line.startswith("- **") or "：" not in line:
                continue
            label, _, rest = line.partition("：")
            label = label.removeprefix("- **").removesuffix("**").strip()
            if label == "核磁文本":
                nmr_at = i + 1
                break
            fields[label] = _unwrap(rest.strip())
        if nmr_at is None:
            raise ValueError(f"{sample_id} 缺少核磁文本")
        nmr_lines: list[str] = []
        for line in lines[nmr_at:]:
            if line.startswith("#"):
                break
            nmr_lines.append(line)
        samples.append(
            Sample(
                sample_id=sample_id,
                group=group,
                sample_type=fields.get("样本类型", ""),
                correct_smiles=fields.get("正确结构（SMILES）", ""),
                negative_smiles=fields.get("负样本结构（SMILES）", ""),
                quality=_strip_ticks(fields.get("质量指标", "")),
                nmr_text="\n".join(nmr_lines).strip(),
            )
        )
    return title, preamble, samples


def _unwrap(value: str) -> str:
    if value.startswith("`") and value.endswith("`") and value.count("`") == 2:
        return value[1:-1]
    return value


def _strip_ticks(value: str) -> str:
    return value.replace("`", "")


def render_mol_png(smiles: str, width: int = 640, height: int = 460) -> bytes | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        rdMolDraw2D.PrepareMolForDrawing(mol)
    except Exception:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.bondLineWidth = 1.8
    options.padding = 0.05
    options.clearBackground = True
    try:
        options.backgroundColour = (1, 1, 1)
    except Exception:
        pass
    try:
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
    except Exception:
        return None
    return drawer.GetDrawingText()


def _split_long_token(token: str, font: str, size: float, width: float) -> list[str]:
    """Break a space-free string such as SMILES so it stays inside `width`."""
    lines: list[str] = []
    buf = ""
    for ch in token:
        trial = buf + ch
        if buf and stringWidth(trial, font, size) > width:
            lines.append(buf)
            buf = ch
        else:
            buf = trial
    if buf:
        lines.append(buf)
    return lines or [""]


def wrap_text(text: str, font: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if para.strip() == "":
            lines.append("")
            continue
        for line in simpleSplit(para, font, size, width) or [""]:
            if line and stringWidth(line, font, size) > width:
                lines.extend(_split_long_token(line, font, size, width))
            else:
                lines.append(line)
    return lines


def draw_wrapped(
    c: canvas.Canvas,
    text: str,
    font: str,
    size: float,
    leading: float,
    x: float,
    top: float,
    width: float,
    color: Color = INK,
) -> float:
    """Draw text below `top` and return the y of the last baseline."""
    c.setFillColor(color)
    c.setFont(font, size)
    y = top - size
    for line in wrap_text(text, font, size, width):
        if line:
            c.drawString(x, y, line)
        y -= leading
    return y + leading


def _fit_image(png: bytes, max_w: float, max_h: float) -> tuple[ImageReader, float, float]:
    image = ImageReader(io.BytesIO(png))
    iw, ih = image.getSize()
    scale = min(max_w / iw, max_h / ih)
    return image, iw * scale, ih * scale


def draw_structure_panel(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    smiles: str,
    png: bytes | None,
    accent: Color,
) -> None:
    c.saveState()
    clip = c.beginPath()
    clip.roundRect(x, y, w, h, 8)
    c.clipPath(clip, stroke=0, fill=0)
    c.setFillColor(PANEL_BG)
    c.rect(x, y, w, h, fill=1, stroke=0)
    c.setFillColor(accent)
    c.rect(x, y + h - 26, w, 26, fill=1, stroke=0)

    c.setFillColor(white)
    c.setFont("NotoSans-Bold", 11)
    c.drawString(x + 10, y + h - 18, title)

    text_width = w - 20
    c.setFillColor(MUTED)
    c.setFont("NotoSans", 8)
    c.drawString(x + 10, y + h - 40, "SMILES")
    smiles_bottom = draw_wrapped(
        c, smiles, "NotoMono", 7.5, 10, x + 10, y + h - 46, text_width, INK
    )

    img_top = smiles_bottom - 6
    img_bottom = y + 8
    img_h = img_top - img_bottom
    img_w = text_width
    if png is None:
        if img_h >= 24:
            c.setFillColor(HexColor("#FEF2F2"))
            c.roundRect(x + 10, img_bottom, img_w, img_h, 4, fill=1, stroke=0)
            c.setFillColor(NEGATIVE_COLOR)
            c.setFont("NotoSans", 9)
            msg = "RDKit 无法解析该 SMILES"
            c.drawCentredString(x + w / 2, img_bottom + img_h / 2 - 4, msg)
    elif img_h >= 40:
        image, dw, dh = _fit_image(png, img_w, img_h)
        c.drawImage(
            image,
            x + 10 + (img_w - dw) / 2,
            img_top - dh,
            dw,
            dh,
            mask="auto",
            preserveAspectRatio=True,
        )
    c.restoreState()

    c.setStrokeColor(accent)
    c.setLineWidth(1.3)
    c.roundRect(x, y, w, h, 8, fill=0, stroke=1)


def draw_sample_page(
    c: canvas.Canvas,
    sample: Sample,
    index: int,
    total: int,
    images: dict[str, bytes | None],
    *,
    first_of_group: bool,
) -> None:
    group_name, group_color = GROUP_STYLE.get(sample.group, (sample.group, INK))
    if first_of_group:
        c.bookmarkPage(f"group-{sample.group}")
        c.addOutlineEntry(group_name, f"group-{sample.group}", level=0, closed=0)
    c.bookmarkPage(sample.sample_id)
    c.addOutlineEntry(sample.sample_id, sample.sample_id, level=1, closed=0)

    c.setFillColor(group_color)
    c.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)

    c.setFillColor(group_color)
    c.setFont("NotoSans-Bold", 16)
    c.drawString(MARGIN, PAGE_H - 34, sample.sample_id)
    c.setFillColor(MUTED)
    c.setFont("NotoSans", 9)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 30, f"{index} / {total}")

    c.setFillColor(INK)
    c.setFont("NotoSans", 10)
    c.drawString(MARGIN, PAGE_H - 54, f"样本类型    {sample.sample_type}")
    c.setFillColor(group_color)
    c.setFont("NotoSans", 9)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 54, group_name)

    y = PAGE_H - 68
    if sample.quality:
        quality_lines = wrap_text(sample.quality, "NotoSans", 8, PAGE_W - 2 * MARGIN - 16)
        qh = 10 + len(quality_lines) * 11
        c.setFillColor(HexColor("#F3F4F6"))
        c.roundRect(MARGIN, y - qh, PAGE_W - 2 * MARGIN, qh, 4, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("NotoSans", 8)
        qy = y - 14
        for line in quality_lines:
            c.drawString(MARGIN + 8, qy, line)
            qy -= 11
        y -= qh + 10
    else:
        y -= 8

    nmr_width = PAGE_W - 2 * MARGIN - 16
    nmr_lines = wrap_text(sample.nmr_text, "NotoMono", 8, nmr_width)
    nmr_h = 22 + len(nmr_lines) * 11 + 8
    c.setFillColor(NMR_BG)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.8)
    c.roundRect(MARGIN, y - nmr_h, PAGE_W - 2 * MARGIN, nmr_h, 6, fill=1, stroke=1)
    c.setFillColor(group_color)
    c.setFont("NotoSans-Bold", 10)
    c.drawString(MARGIN + 8, y - 16, "核磁文本")
    c.setFillColor(INK)
    c.setFont("NotoMono", 8)
    ny = y - 32
    for line in nmr_lines:
        if line:
            c.drawString(MARGIN + 8, ny, line)
        ny -= 11

    panel_top = y - nmr_h - 12
    panel_bottom = 36
    panel_h = panel_top - panel_bottom
    gap = 12
    panel_w = (PAGE_W - 2 * MARGIN - gap) / 2
    draw_structure_panel(
        c,
        MARGIN,
        panel_bottom,
        panel_w,
        panel_h,
        "正确结构",
        sample.correct_smiles,
        images.get(sample.correct_smiles),
        CORRECT_COLOR,
    )
    draw_structure_panel(
        c,
        MARGIN + panel_w + gap,
        panel_bottom,
        panel_w,
        panel_h,
        "负样本结构",
        sample.negative_smiles,
        images.get(sample.negative_smiles),
        NEGATIVE_COLOR,
    )

    c.setFillColor(MUTED)
    c.setFont("NotoSans", 8)
    c.drawString(MARGIN, 18, group_name)
    c.drawRightString(PAGE_W - MARGIN, 18, sample.sample_id)
    c.showPage()


def draw_cover(
    c: canvas.Canvas,
    title: str,
    preamble: list[str],
    samples: list[Sample],
) -> None:
    c.bookmarkPage("cover")
    c.addOutlineEntry(title, "cover", level=0, closed=0)

    c.setFillColor(HexColor("#111827"))
    c.rect(0, PAGE_H - 120, PAGE_W, 120, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("NotoSans-Bold", 20)
    # Title can be long; wrap inside the banner.
    title_lines = wrap_text(title, "NotoSans-Bold", 18, PAGE_W - 2 * MARGIN)
    ty = PAGE_H - 48
    c.setFont("NotoSans-Bold", 18)
    for line in title_lines[:3]:
        c.drawString(MARGIN, ty, line)
        ty -= 24
    c.setFont("NotoSans", 11)
    c.drawString(MARGIN, PAGE_H - 100, f"共 {len(samples)} 条，每页 1 条")

    y = PAGE_H - 156
    c.setFillColor(INK)
    c.setFont("NotoSans-Bold", 12)
    c.drawString(MARGIN, y, "来源")
    y -= 18
    c.setFont("NotoSans", 9)
    c.setFillColor(MUTED)
    for item in preamble:
        for line in wrap_text(item, "NotoSans", 9, PAGE_W - 2 * MARGIN - 12):
            c.drawString(MARGIN + 8, y, line)
            y -= 13
        y -= 4

    y -= 10
    c.setFillColor(INK)
    c.setFont("NotoSans-Bold", 12)
    c.drawString(MARGIN, y, "四类样本")
    y -= 16
    card_w = (PAGE_W - 2 * MARGIN - 12) / 2
    card_h = 72
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.group] = counts.get(sample.group, 0) + 1
    keys = list(GROUP_STYLE)
    for i, key in enumerate(keys):
        name, color = GROUP_STYLE[key]
        col = i % 2
        row = i // 2
        x = MARGIN + col * (card_w + 12)
        cy = y - (row + 1) * card_h - row * 10
        c.setFillColor(color)
        c.roundRect(x, cy, card_w, card_h, 6, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("NotoSans-Bold", 13)
        c.drawString(x + 14, cy + 42, name)
        c.setFont("NotoSans", 11)
        c.drawString(x + 14, cy + 20, f"{counts.get(key, 0)} 条")

    c.showPage()


def build_pdf(md_path: Path, pdf_path: Path) -> None:
    register_fonts()
    title, preamble, samples = parse_markdown(md_path)
    if len(samples) != 120:
        raise SystemExit(f"期望 120 条样本，实际解析到 {len(samples)} 条：{md_path}")
    missing = [s.sample_id for s in samples if not (s.nmr_text and s.correct_smiles and s.negative_smiles and s.sample_type)]
    if missing:
        raise SystemExit(f"字段不完整：{', '.join(missing)}")

    smiles = {s.correct_smiles for s in samples} | {s.negative_smiles for s in samples}
    images = {smiles_text: render_mol_png(smiles_text) for smiles_text in smiles}
    failed = [s for s, png in images.items() if png is None]
    if failed:
        print(f"warning: {len(failed)} SMILES 无法绘图")
        for item in failed:
            print(" ", item)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.setTitle(title)
    c.setAuthor("SpecMaster")
    draw_cover(c, title, preamble, samples)
    seen_groups: set[str] = set()
    for i, sample in enumerate(samples, start=1):
        first = sample.group not in seen_groups
        seen_groups.add(sample.group)
        draw_sample_page(c, sample, i, len(samples), images, first_of_group=first)
    c.save()
    print(f"wrote {pdf_path} ({len(samples)} sample pages + cover)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_MD)
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()
    build_pdf(args.input, args.output)


if __name__ == "__main__":
    main()
