#!/usr/bin/env python3
"""Generate an SVG summary table: Paper results vs Our results for all models."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path


COLORS = {
    "bg": "#ffffff",
    "header": "#0f766e",
    "header_text": "#ffffff",
    "row_even": "#f0fdf4",
    "row_odd": "#ffffff",
    "ink": "#172026",
    "muted": "#65737e",
    "border": "#d1d5db",
    "accent": "#0f766e",
    "better": "#15803d",
    "worse": "#dc2626",
    "neutral": "#65737e",
}

ROWS = [
    # (model/dataset, metric, paper_fft, our_fft, paper_lora, our_lora)
    ("ViT / CIFAR-10", "Accuracy", None, "98.68", None, "98.30"),
    ("RoBERTa / SST-2", "Accuracy", "94.8", "93.9", "95.1", "94.0"),
    ("RoBERTa / MNLI", "Accuracy", "87.6", "87.65", "87.5", "85.9"),
    ("GPT-2 Small / E2E", "BLEU", "53.68", "53.68", None, "52.98"),
    ("GPT-2 Small / E2E", "NIST", "6.0225", "6.0225", None, "3.1081"),
    ("GPT-2 Small / E2E", "METEOR", "69.58", "69.58", None, "48.99"),
    ("GPT-2 Small / E2E", "ROUGE-L", "62.55", "62.55", None, "62.84"),
    ("GPT-2 Small / E2E", "CIDEr", "1.17", "1.17", None, "1.0428"),
    ("GPT-2 Medium / E2E", "BLEU", "68.2", None, "54.11", "53.68"),
    ("GPT-2 Medium / E2E", "NIST", "8.62", None, "6.0225", "6.0468"),
    ("GPT-2 Medium / E2E", "METEOR", "46.2", None, "69.1", "65.18"),
    ("GPT-2 Medium / E2E", "ROUGE-L", "71.0", None, "62.03", "63.33"),
    ("GPT-2 Medium / E2E", "CIDEr", "2.47", None, "1.2261", "1.097"),
]


def t(x, y, text, size=16, weight=400, fill=COLORS["ink"], anchor="start"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        f"{escape(str(text))}</text>"
    )


def r(x, y, w, h, fill, rx=0, stroke="none"):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def build_svg() -> str:
    col_widths = [220, 80, 90, 90, 90, 90]
    total_w = sum(col_widths) + 80
    row_h = 36
    header_h = 44
    title_h = 70
    n_rows = len(ROWS)
    total_h = title_h + header_h + n_rows * row_h + 40

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">',
        r(0, 0, total_w, total_h, COLORS["bg"]),
        t(40, 42, "Results Summary: Paper vs. Our Re-implementation", 24, 850),
    ]

    headers = ["Model / Dataset", "Metric", "Paper (FFT)", "Ours (FFT)", "Paper (LoRA)", "Ours (LoRA)"]
    hx = 40
    hy = title_h
    parts.append(r(hx, hy, sum(col_widths), header_h, COLORS["header"], 6))
    cx = hx
    for i, (hdr, cw) in enumerate(zip(headers, col_widths)):
        parts.append(t(cx + 10, hy + 28, hdr, 14, 700, COLORS["header_text"]))
        cx += cw

    for row_idx, (model, metric, p_fft, o_fft, p_lora, o_lora) in enumerate(ROWS):
        ry = hy + header_h + row_idx * row_h
        bg = COLORS["row_even"] if row_idx % 2 == 0 else COLORS["row_odd"]
        rx_val = 6 if row_idx == n_rows - 1 else 0
        parts.append(r(hx, ry, sum(col_widths), row_h, bg, rx_val))
        parts.append(f'<line x1="{hx}" y1="{ry + row_h}" x2="{hx + sum(col_widths)}" y2="{ry + row_h}" stroke="{COLORS["border"]}" stroke-width="0.5"/>')

        vals = [model, metric, p_fft or "—", o_fft or "—", p_lora or "—", o_lora or "—"]
        cx = hx
        for i, (val, cw) in enumerate(zip(vals, col_widths)):
            weight = 700 if i == 0 else 400
            size = 13 if i == 0 else 14
            parts.append(t(cx + 10, ry + 24, val, size, weight))
            cx += cw

    parts.append(r(hx, hy, sum(col_widths), header_h + n_rows * row_h, "none", 6, COLORS["border"]))
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/summary_table.svg"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
