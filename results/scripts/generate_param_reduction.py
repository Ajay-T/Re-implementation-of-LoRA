#!/usr/bin/env python3
"""Generate a parameter reduction visualization.

Shows the dramatic reduction in trainable parameters when using LoRA,
with donut-style charts for each model.
"""

from __future__ import annotations

import argparse
import math
from html import escape
from pathlib import Path

COLORS = {
    "bg": "#fbfbf8",
    "panel": "#ffffff",
    "ink": "#172026",
    "muted": "#65737e",
    "border": "#d8ded9",
    "frozen": "#e5e7eb",
    "lora": "#0f766e",
    "lora_light": "#99f6e4",
}

MODELS = [
    ("ViT-B/16\nCIFAR-10", 85_806_346, 302_602),
    ("RoBERTa-base\nSST-2", 124_647_170, 887_042),
    ("RoBERTa-base\nMNLI", 124_647_170, 887_811),
    ("GPT-2 Small\nE2E NLG", 124_439_808, 147_456),
    ("GPT-2 Medium\nE2E NLG", 124_439_808, 393_216),
]


def fmt(v):
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return str(v)


def t(x, y, text, size=16, weight=400, fill=COLORS["ink"], anchor="middle", extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>'
        f"{escape(str(text))}</text>"
    )


def arc_path(cx, cy, r, start_angle, end_angle):
    s_rad = math.radians(start_angle - 90)
    e_rad = math.radians(end_angle - 90)
    x1 = cx + r * math.cos(s_rad)
    y1 = cy + r * math.sin(s_rad)
    x2 = cx + r * math.cos(e_rad)
    y2 = cy + r * math.sin(e_rad)
    large = 1 if (end_angle - start_angle) > 180 else 0
    return f"M {x1:.2f} {y1:.2f} A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f}"


def build_svg() -> str:
    n = len(MODELS)
    donut_r = 52
    donut_inner = 36
    col_w = 180
    W = col_w * n + 80
    H = 300
    start_x = 40 + col_w // 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
        t(W / 2, 30, "Trainable Parameters: LoRA vs. Total Model Parameters", 18, 850),
    ]

    for i, (label, total, lora) in enumerate(MODELS):
        cx = start_x + i * col_w
        cy = 120
        pct = 100.0 * lora / total
        angle = 360.0 * lora / total

        # Full circle (frozen params)
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{donut_r}" fill="{COLORS["frozen"]}" stroke="none"/>')
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{donut_inner}" fill="{COLORS["panel"]}" stroke="none"/>')

        # LoRA slice — for very small percentages, draw a dot instead
        if angle > 1:
            path = arc_path(cx, cy, donut_r, 0, max(angle, 3))
            parts.append(f'<path d="{path}" fill="none" stroke="{COLORS["lora"]}" stroke-width="{donut_r - donut_inner}" stroke-linecap="butt"/>')
        # Re-draw inner circle to clean up
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{donut_inner}" fill="{COLORS["panel"]}" stroke="none"/>')

        # Center text
        parts.append(t(cx, cy + 5, f"{pct:.2f}%", 13, 800, COLORS["lora"]))

        # Labels below
        lines = label.split("\n")
        parts.append(t(cx, cy + donut_r + 20, lines[0], 12, 700))
        if len(lines) > 1:
            parts.append(t(cx, cy + donut_r + 35, lines[1], 11, 400, COLORS["muted"]))

        parts.append(t(cx, cy + donut_r + 52, f"LoRA: {fmt(lora)}", 10, 600, COLORS["lora"]))
        parts.append(t(cx, cy + donut_r + 65, f"Total: {fmt(total)}", 10, 400, COLORS["muted"]))

    # Legend
    lx = 40
    ly = H - 18
    parts.append(f'<rect x="{lx}" y="{ly - 8}" width="12" height="12" rx="2" fill="{COLORS["frozen"]}"/>')
    parts.append(t(lx + 16, ly + 3, "Frozen params", 10, 400, COLORS["muted"], "start"))
    parts.append(f'<rect x="{lx + 110}" y="{ly - 8}" width="12" height="12" rx="2" fill="{COLORS["lora"]}"/>')
    parts.append(t(lx + 126, ly + 3, "LoRA trainable params", 10, 400, COLORS["muted"], "start"))

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/param_reduction.svg"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
