#!/usr/bin/env python3
"""Generate a poster-ready performance comparison graphic.

This figure complements the cost graphic. It uses dumbbell plots to show that
LoRA often stays close to full fine-tuning in task performance while making
larger gaps visible where they occur.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path


COLORS = {
    "bg": "#fbfbf8",
    "panel": "#ffffff",
    "ink": "#172026",
    "muted": "#65737e",
    "grid": "#d9dfdc",
    "fft": "#6b7280",
    "lora": "#0f766e",
    "line": "#cbd5d1",
    "win": "#15803d",
    "loss": "#b45309",
}


@dataclass(frozen=True)
class ClassificationResult:
    task: str
    source: str
    fft: float
    lora: float


@dataclass(frozen=True)
class GenerationResult:
    model: str
    metric: str
    fft: float
    lora: float


CLASSIFICATION_RESULTS = [
    ClassificationResult("SST-2", "Paper", 94.8, 95.1),
    ClassificationResult("SST-2", "Ours", 93.9, 94.0),
    ClassificationResult("MNLI", "Paper", 87.6, 87.5),
    # Poster table reports FFT as 0.877 matched / 0.876 mismatched and LoRA as 0.859.
    # Use the average FFT accuracy so each row has one comparable scalar.
    ClassificationResult("MNLI", "Ours", (87.7 + 87.6) / 2.0, 85.9),
    ClassificationResult("CIFAR-10", "Ours", 98.68, 98.30),
]


GENERATION_RESULTS = [
    GenerationResult("GPT-2 Small", "BLEU", 53.68, 52.98),
    GenerationResult("GPT-2 Small", "NIST", 6.0225, 3.1081),
    GenerationResult("GPT-2 Small", "METEOR", 69.58, 48.99),
    GenerationResult("GPT-2 Small", "ROUGE-L", 62.55, 62.84),
    GenerationResult("GPT-2 Small", "CIDEr", 1.17, 1.0428),
    GenerationResult("GPT-2 Medium", "BLEU", 68.2, 53.68),
    GenerationResult("GPT-2 Medium", "NIST", 8.62, 6.0468),
    GenerationResult("GPT-2 Medium", "METEOR", 46.2, 65.18),
    GenerationResult("GPT-2 Medium", "ROUGE-L", 71.0, 63.33),
    GenerationResult("GPT-2 Medium", "CIDEr", 1.2261, 1.097),
]


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 18,
    weight: int | str = 400,
    fill: str = COLORS["ink"],
    anchor: str = "start",
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}" {extra}>{escape(text)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, rx: float = 0, stroke: str = "none") -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}"/>'
    )


def scale(value: float, lo: float, hi: float, x0: float, x1: float) -> float:
    return x0 + (value - lo) * (x1 - x0) / (hi - lo)


def draw_legend(x: float, y: float) -> list[str]:
    return [
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{COLORS["fft"]}"/>',
        svg_text(x + 20, y + 6, "FFT", 22, 700, COLORS["ink"]),
        f'<circle cx="{x + 210:.1f}" cy="{y:.1f}" r="10" fill="{COLORS["lora"]}"/>',
        svg_text(x + 230, y + 6, "LoRA", 22, 700, COLORS["ink"]),
    ]


def draw_classification_panel(x: float, y: float, w: float, h: float) -> list[str]:
    parts: list[str] = [rect(x, y, w, h, COLORS["panel"], 12, "#d8ded9")]
    parts.append(svg_text(x + 34, y + 54, "Classification Accuracy", 32, 850))

    label_x = x + 44
    source_x = x + 202
    axis_x0 = x + 315
    axis_x1 = x + w - 210
    axis_y = y + 132
    lo, hi = 84.0, 100.0

    for tick in [85, 90, 95, 100]:
        tx = scale(tick, lo, hi, axis_x0, axis_x1)
        parts.append(f'<line x1="{tx:.1f}" y1="{axis_y:.1f}" x2="{tx:.1f}" y2="{y + h - 54:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(svg_text(tx, axis_y - 14, f"{tick}%", 16, 700, COLORS["muted"], "middle"))

    row_y = y + 184
    row_gap = 78
    for idx, result in enumerate(CLASSIFICATION_RESULTS):
        cy = row_y + idx * row_gap
        fft_x = scale(result.fft, lo, hi, axis_x0, axis_x1)
        lora_x = scale(result.lora, lo, hi, axis_x0, axis_x1)
        left, right = sorted([fft_x, lora_x])
        parts.append(svg_text(label_x, cy + 8, result.task, 22, 800))
        source_fill = COLORS["muted"] if result.source == "Paper" else COLORS["lora"]
        parts.append(svg_text(source_x, cy + 8, result.source, 18, 800, source_fill))
        parts.append(f'<line x1="{axis_x0:.1f}" y1="{cy:.1f}" x2="{axis_x1:.1f}" y2="{cy:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<line x1="{left:.1f}" y1="{cy:.1f}" x2="{right:.1f}" y2="{cy:.1f}" stroke="{COLORS["line"]}" stroke-width="8" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{fft_x:.1f}" cy="{cy:.1f}" r="14" fill="{COLORS["fft"]}" stroke="#ffffff" stroke-width="4"/>')
        parts.append(f'<circle cx="{lora_x:.1f}" cy="{cy:.1f}" r="14" fill="{COLORS["lora"]}" stroke="#ffffff" stroke-width="4"/>')
        parts.append(svg_text(axis_x1 + 28, cy + 8, f"{result.fft:.1f}% -> {result.lora:.1f}%", 18, 800, COLORS["ink"]))

    return parts


def draw_generation_panel(x: float, y: float, w: float, h: float, model: str) -> list[str]:
    parts: list[str] = [rect(x, y, w, h, COLORS["panel"], 12, "#d8ded9")]
    parts.append(svg_text(x + 34, y + 54, model, 32, 850))

    gx = x + 44
    top_y = y + 140
    row_gap = 84
    label_w = 104

    rows = [r for r in GENERATION_RESULTS if r.model == model]
    axis_x0 = gx + label_w
    axis_x1 = x + w - 245
    lo, hi = 0.0, 75.0

    for tick in [0, 25, 50, 75]:
        tx = scale(tick, lo, hi, axis_x0, axis_x1)
        parts.append(f'<line x1="{tx:.1f}" y1="{top_y - 34:.1f}" x2="{tx:.1f}" y2="{top_y + (len(rows) - 1) * row_gap + 34:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(svg_text(tx, top_y - 48, f"{tick}", 15, 700, COLORS["muted"], "middle"))

    for row_idx, result in enumerate(rows):
        cy = top_y + row_idx * row_gap
        fft_x = scale(result.fft, lo, hi, axis_x0, axis_x1)
        lora_x = scale(result.lora, lo, hi, axis_x0, axis_x1)
        left, right = sorted([fft_x, lora_x])

        parts.append(svg_text(gx, cy + 8, result.metric, 20, 800))
        parts.append(f'<line x1="{axis_x0:.1f}" y1="{cy:.1f}" x2="{axis_x1:.1f}" y2="{cy:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(f'<line x1="{left:.1f}" y1="{cy:.1f}" x2="{right:.1f}" y2="{cy:.1f}" stroke="{COLORS["line"]}" stroke-width="7" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{fft_x:.1f}" cy="{cy:.1f}" r="12" fill="{COLORS["fft"]}" stroke="#ffffff" stroke-width="4"/>')
        parts.append(f'<circle cx="{lora_x:.1f}" cy="{cy:.1f}" r="12" fill="{COLORS["lora"]}" stroke="#ffffff" stroke-width="4"/>')
        parts.append(svg_text(axis_x1 + 22, cy + 7, f"{result.fft:g} -> {result.lora:g}", 17, 800, COLORS["ink"]))

    return parts


def build_svg() -> str:
    width, height = 1120, 1950
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, COLORS["bg"]),
        svg_text(80, 78, "Task Performance: FFT vs. LoRA", 44, 850),
    ]
    parts.extend(draw_legend(805, 70))
    parts.extend(draw_classification_panel(70, 130, 980, 555))
    parts.extend(draw_generation_panel(70, 745, 980, 530, "GPT-2 Small"))
    parts.extend(draw_generation_panel(70, 1335, 980, 530, "GPT-2 Medium"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lora_fft_performance_dumbbell.svg"),
        help="Path to write the generated SVG.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
