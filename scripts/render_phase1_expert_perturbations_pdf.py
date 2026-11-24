#!/usr/bin/env python3
"""Render the Phase 1 expert-perturbation review markdown as a one-sample-per-page PDF."""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_v6_4_negative_samples_pdf import (  # noqa: E402
    CORRECT_COLOR,
    INK,
    LINE,
    MARGIN,
    MUTED,
    NEGATIVE_COLOR,
    NMR_BG,
    PAGE_H,
    PAGE_W,
    draw_structure_panel,
    register_fonts,
    render_mol_png,
    wrap_text,
)

DEFAULT_MD = ROOT / "docs" / "phase1_expert_perturbations_review30.md"
DEFAULT_PDF = ROOT / "docs" / "phase1_expert_perturbations_review30.pdf"

GROUP_STYLE = {
    "halogen_swap": ("卤素替换", HexColor("#1D4ED8")),
    "ring_chalcogen_swap": ("环杂原子替换", HexColor("#047857")),
    "positional_isomer": ("位置异构", HexColor("#B45309")),
    "connectivity_isomer": ("连接异构", HexColor("#7E22CE")),
}
SUBTYPE_LABEL = {
    "Cl_to_Br": "Cl→Br",
    "ring_S_to_O": "环 S→O",
    "ring_O_to_S": "环 O→S",
    "aromatic_substituent_shift": "芳环取代位移",
    "mined_same_formula_attachment_isomer": "同分子式连接异构",
}


@dataclass
class Sample:
    sample_id: str
    group: str
    subtype: str
    sample_type: str
    correct_smiles: str
    negative_smiles: str
    evidence: str
    nmr_text: str

    @property
    def group_label(self) -> str:
        name, _color = GROUP_STYLE.get(self.group, (self.group, INK))
        detail = SUBTYPE_LABEL.get(self.subtype, self.subtype)
        return f"{name} · {detail}" if detail else name

    @property
    def group_color(self):
        return GROUP_STYLE.get(self.group, (self.group, INK))[1]


def _unwrap(value: str) -> str:
    if value.startswith("`") and value.endswith("`") and value.count("`") == 2:
        return value[1:-1]
    return value


def _strip_ticks(value: str) -> str:
    return value.replace("`", "")


def parse_markdown(path: Path) -> tuple[str, list[str], list[Sample]]:
    text = path.read_text(encoding="utf-8")
    title = "Phase 1 专家核验"
    notes: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            break
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            notes.append(_strip_ticks(stripped[2:].strip()))

    samples: list[Sample] = []
    for chunk in re.split(r"(?=^## )", text, flags=re.M):
        if not chunk.startswith("## "):
            continue
        lines = chunk.splitlines()
        heading = lines[0][3:].strip()
        sample_id, _, _rest = heading.partition(". ")
        sample_id = sample_id.strip()
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
        sample_type = fields.get("样本类型", "")
        parts = sample_type.split("/")
        group = parts[2] if len(parts) >= 3 else "other"
        subtype = parts[3] if len(parts) >= 4 else ""
        samples.append(
            Sample(
                sample_id=sample_id,
                group=group,
                subtype=subtype,
                sample_type=sample_type,
                correct_smiles=fields.get("正确结构（SMILES）", ""),
                negative_smiles=fields.get("负样本结构（SMILES）", ""),
                evidence=_strip_ticks(fields.get("筛选证据", "")),
                nmr_text="\n".join(nmr_lines).strip(),
            )
        )
    return title, notes, samples


def draw_sample_page(
    c: canvas.Canvas,
    sample: Sample,
    index: int,
    total: int,
    images: dict[str, bytes | None],
    *,
    first_of_group: bool,
) -> None:
    if first_of_group:
        name = GROUP_STYLE.get(sample.group, (sample.group, INK))[0]
        c.bookmarkPage(f"group-{sample.group}")
        c.addOutlineEntry(name, f"group-{sample.group}", level=0, closed=0)
    c.bookmarkPage(sample.sample_id)
    c.addOutlineEntry(f"{sample.sample_id}  {sample.group_label}", sample.sample_id, level=1, closed=0)

    c.setFillColor(sample.group_color)
    c.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)

    c.setFillColor(sample.group_color)
    c.setFont("NotoSans-Bold", 16)
    c.drawString(MARGIN, PAGE_H - 34, sample.sample_id)
    id_width = c.stringWidth(sample.sample_id, "NotoSans-Bold", 16)
    c.setFont("NotoSans", 11)
    c.drawString(MARGIN + id_width + 10, PAGE_H - 32, sample.group_label)
    c.setFillColor(MUTED)
    c.setFont("NotoSans", 9)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 30, f"{index} / {total}")

    y = PAGE_H - 48
    type_lines = wrap_text(f"样本类型    {sample.sample_type}", "NotoSans", 9, PAGE_W - 2 * MARGIN)
    c.setFillColor(INK)
    c.setFont("NotoSans", 9)
    for line in type_lines:
        c.drawString(MARGIN, y, line)
        y -= 12
    y -= 4

    if sample.evidence:
        evidence_lines = wrap_text(sample.evidence, "NotoSans", 8, PAGE_W - 2 * MARGIN - 16)
        box_h = 10 + len(evidence_lines) * 11
        c.setFillColor(HexColor("#F3F4F6"))
        c.roundRect(MARGIN, y - box_h, PAGE_W - 2 * MARGIN, box_h, 4, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("NotoSans", 8)
        ey = y - 14
        for line in evidence_lines:
            c.drawString(MARGIN + 8, ey, line)
            ey -= 11
        y -= box_h + 10

    nmr_width = PAGE_W - 2 * MARGIN - 16
    nmr_lines = wrap_text(sample.nmr_text, "NotoMono", 8, nmr_width)
    nmr_h = 22 + len(nmr_lines) * 11 + 8
    c.setFillColor(NMR_BG)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.8)
    c.roundRect(MARGIN, y - nmr_h, PAGE_W - 2 * MARGIN, nmr_h, 6, fill=1, stroke=1)
    c.setFillColor(sample.group_color)
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
        c, MARGIN, panel_bottom, panel_w, panel_h,
        "正确结构", sample.correct_smiles, images.get(sample.correct_smiles), CORRECT_COLOR,
    )
    draw_structure_panel(
        c, MARGIN + panel_w + gap, panel_bottom, panel_w, panel_h,
        "负样本结构", sample.negative_smiles, images.get(sample.negative_smiles), NEGATIVE_COLOR,
    )

    c.setFillColor(MUTED)
    c.setFont("NotoSans", 8)
    c.drawString(MARGIN, 18, sample.group_label)
    c.drawRightString(PAGE_W - MARGIN, 18, sample.sample_id)
    c.showPage()


def draw_cover(c: canvas.Canvas, title: str, notes: list[str], samples: list[Sample]) -> None:
    c.bookmarkPage("cover")
    c.addOutlineEntry(title, "cover", level=0, closed=0)

    c.setFillColor(HexColor("#111827"))
    c.rect(0, PAGE_H - 110, PAGE_W, 110, fill=1, stroke=0)
    c.setFillColor(white)
    title_lines = wrap_text(title, "NotoSans-Bold", 16, PAGE_W - 2 * MARGIN)
    ty = PAGE_H - 42
    c.setFont("NotoSans-Bold", 16)
    for line in title_lines[:2]:
        c.drawString(MARGIN, ty, line)
        ty -= 22
    c.setFont("NotoSans", 11)
    c.drawString(MARGIN, PAGE_H - 92, f"共 {len(samples)} 条，每页 1 条")

    y = PAGE_H - 140
    c.setFillColor(INK)
    c.setFont("NotoSans-Bold", 12)
    c.drawString(MARGIN, y, "说明")
    y -= 18
    c.setFillColor(MUTED)
    c.setFont("NotoSans", 9)
    for item in notes:
        for line in wrap_text(item, "NotoSans", 9, PAGE_W - 2 * MARGIN - 12):
            c.drawString(MARGIN + 8, y, line)
            y -= 13
        y -= 6

    y -= 8
    c.setFillColor(INK)
    c.setFont("NotoSans-Bold", 12)
    c.drawString(MARGIN, y, "四类微扰")
    y -= 16
    card_w = (PAGE_W - 2 * MARGIN - 12) / 2
    card_h = 72
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.group] = counts.get(sample.group, 0) + 1
    for i, key in enumerate(GROUP_STYLE):
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
    title, notes, samples = parse_markdown(md_path)
    if len(samples) != 30:
        raise SystemExit(f"期望 30 条样本，实际解析到 {len(samples)} 条：{md_path}")
    missing = [
        s.sample_id
        for s in samples
        if not (s.nmr_text and s.correct_smiles and s.negative_smiles and s.sample_type and s.evidence)
    ]
    if missing:
        raise SystemExit(f"字段不完整：{', '.join(missing)}")

    smiles = {s.correct_smiles for s in samples} | {s.negative_smiles for s in samples}
    images = {text: render_mol_png(text) for text in smiles}
    failed = [text for text, png in images.items() if png is None]
    if failed:
        print(f"warning: {len(failed)} SMILES 无法绘图")
        for item in failed:
            print(" ", item)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.setTitle(title)
    c.setAuthor("SpecMaster")
    draw_cover(c, title, notes, samples)
    seen: set[str] = set()
    for i, sample in enumerate(samples, start=1):
        first = sample.group not in seen
        seen.add(sample.group)
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
