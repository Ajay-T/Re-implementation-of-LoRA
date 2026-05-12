#!/usr/bin/env python3
"""Generate an efficiency scatter plot: accuracy/quality vs resource cost.

Shows that LoRA achieves comparable task performance while using dramatically
fewer resources (trainable parameters, VRAM, time).
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
    "grid": "#e5e7eb",
    "fft": "#6b7280",
    "lora": "#0f766e",
    "fft_fill": "#d1d5db",
    "lora_fill": "#99f6e4",
    "border": "#d8ded9",
}

# (label, fft_vram_mb, lora_vram_mb, fft_perf, lora_perf, perf_label)
POINTS = [
    ("ViT\nCIFAR-10", 3522.9, 2296.7, 98.68, 98.30, "Accuracy"),
    ("RoBERTa\nSST-2", 2483.5, 1317.5, 93.9, 94.0, "Accuracy"),
    ("RoBERTa\nMNLI", 14708.1, 1317.5, 87.65, 85.9, "Accuracy"),
    ("GPT-2 S\nE2E", 7751.6, 3624.4, 62.55, 62.84, "ROUGE-L"),
    ("GPT-2 M\nE2E", 7134.2, 3624.4, 71.0, 63.33, "ROUGE-L"),
]


def t(x, y, text, size=16, weight=400, fill=COLORS["ink"], anchor="start", extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>'
        f"{escape(str(text))}</text>"
    )


def build_svg() -> str:
    W, H = 900, 620
    pad_l, pad_r, pad_t, pad_b = 100, 60, 80, 70
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b

    x_min, x_max = 0, 16000
    y_min, y_max = 60, 100

    def sx(v):
        return pad_l + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v):
        return pad_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="{COLORS["panel"]}" stroke="{COLORS["border"]}"/>',
        t(W / 2, 36, "Efficiency Frontier: Task Performance vs. Peak VRAM", 20, 850, anchor="middle"),
        t(W / 2, 56, "Arrows show the shift from FFT (gray) to LoRA (teal) for each experiment", 13, 400, COLORS["muted"], "middle"),
    ]

    # Grid
    for xv in range(0, 16001, 4000):
        px = sx(xv)
        parts.append(f'<line x1="{px:.1f}" y1="{pad_t}" x2="{px:.1f}" y2="{pad_t + plot_h}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        label = f"{xv / 1000:.0f}GB" if xv >= 1000 else str(xv)
        parts.append(t(px, H - pad_b + 20, f"{xv:,} MB", 11, 600, COLORS["muted"], "middle"))

    for yv in range(60, 101, 10):
        py = sy(yv)
        parts.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{pad_l + plot_w}" y2="{py:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(t(pad_l - 10, py + 4, f"{yv}", 12, 600, COLORS["muted"], "end"))

    parts.append(t(W / 2, H - 8, "Peak VRAM (MB)", 13, 700, COLORS["muted"], "middle"))
    parts.append(t(16, H / 2, "Task Performance", 13, 700, COLORS["muted"], "middle", 'transform="rotate(-90,16,' + f'{H/2:.0f})"'))

    # Arrows and points
    for label, fft_vram, lora_vram, fft_perf, lora_perf, _ in POINTS:
        fx, fy = sx(fft_vram), sy(fft_perf)
        lx, ly = sx(lora_vram), sy(lora_perf)

        # Arrow line
        dx, dy = lx - fx, ly - fy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0:
            ux, uy = dx / dist, dy / dist
            parts.append(
                f'<line x1="{fx:.1f}" y1="{fy:.1f}" x2="{lx - ux * 14:.1f}" y2="{ly - uy * 14:.1f}" '
                f'stroke="{COLORS["lora"]}" stroke-width="2" marker-end="url(#arrow)"/>'
            )

        parts.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="8" fill="{COLORS["fft_fill"]}" stroke="{COLORS["fft"]}" stroke-width="2"/>')
        parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="8" fill="{COLORS["lora_fill"]}" stroke="{COLORS["lora"]}" stroke-width="2"/>')

        name = label.split("\n")[0]
        parts.append(t(lx + 12, ly - 8, name, 11, 700, COLORS["ink"]))
        parts.append(t(lx + 12, ly + 5, label.split("\n")[1], 10, 400, COLORS["muted"]))

    # Arrow marker
    parts.insert(1,
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
        f'<path d="M0,0 L8,4 L0,8 Z" fill="{COLORS["lora"]}"/></marker></defs>'
    )

    # Legend
    lx, ly = pad_l + 10, pad_t + 16
    parts.append(f'<circle cx="{lx}" cy="{ly}" r="6" fill="{COLORS["fft_fill"]}" stroke="{COLORS["fft"]}" stroke-width="2"/>')
    parts.append(t(lx + 12, ly + 4, "Full Fine-Tune", 11, 600))
    parts.append(f'<circle cx="{lx + 110}" cy="{ly}" r="6" fill="{COLORS["lora_fill"]}" stroke="{COLORS["lora"]}" stroke-width="2"/>')
    parts.append(t(lx + 122, ly + 4, "LoRA", 11, 600))

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/efficiency_scatter.svg"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
