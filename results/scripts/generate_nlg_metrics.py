#!/usr/bin/env python3
"""Generate grouped bar chart comparing FFT vs LoRA on NLG metrics (E2E dataset)."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

COLORS = {
    "bg": "#fbfbf8",
    "panel": "#ffffff",
    "ink": "#172026",
    "muted": "#65737e",
    "grid": "#e5e7eb",
    "border": "#d8ded9",
    "fft": "#9ca3af",
    "lora": "#0f766e",
}

# (metric, gpt2s_fft, gpt2s_lora, gpt2m_fft, gpt2m_lora)
METRICS = [
    ("BLEU", 53.68, 52.98, 68.2, 53.68),
    ("NIST", 6.0225, 3.1081, 8.62, 6.0468),
    ("METEOR", 69.58, 48.99, 46.2, 65.18),
    ("ROUGE-L", 62.55, 62.84, 71.0, 63.33),
    ("CIDEr", 1.17, 1.0428, 1.2261, 1.097),
]


def t(x, y, text, size=14, weight=400, fill=COLORS["ink"], anchor="start", extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>'
        f"{escape(str(text))}</text>"
    )


def draw_model_panel(parts, x0, y0, pw, ph, title, metrics_data):
    """metrics_data: list of (metric, fft_val, lora_val)"""
    parts.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="8" fill="{COLORS["panel"]}" stroke="{COLORS["border"]}"/>')
    parts.append(t(x0 + 16, y0 + 26, title, 16, 800))

    n = len(metrics_data)
    bar_area_x = x0 + 80
    bar_area_w = pw - 100
    bar_area_y = y0 + 44
    bar_area_h = ph - 60
    group_w = bar_area_w / n
    bar_w = group_w * 0.32
    gap = group_w * 0.06

    max_val = max(max(f, l) for _, f, l in metrics_data) * 1.15

    for i, (metric, fft, lora) in enumerate(metrics_data):
        gx = bar_area_x + i * group_w
        # Bars
        fft_h = (fft / max_val) * bar_area_h
        lora_h = (lora / max_val) * bar_area_h
        fft_y = bar_area_y + bar_area_h - fft_h
        lora_y = bar_area_y + bar_area_h - lora_h

        parts.append(f'<rect x="{gx + gap:.1f}" y="{fft_y:.1f}" width="{bar_w:.1f}" height="{fft_h:.1f}" rx="3" fill="{COLORS["fft"]}"/>')
        parts.append(f'<rect x="{gx + bar_w + gap * 2:.1f}" y="{lora_y:.1f}" width="{bar_w:.1f}" height="{lora_h:.1f}" rx="3" fill="{COLORS["lora"]}"/>')

        # Value labels
        parts.append(t(gx + gap + bar_w / 2, fft_y - 4, f"{fft:g}", 9, 600, COLORS["muted"], "middle"))
        parts.append(t(gx + bar_w + gap * 2 + bar_w / 2, lora_y - 4, f"{lora:g}", 9, 600, COLORS["lora"], "middle"))

        # Metric label
        parts.append(t(gx + group_w / 2, bar_area_y + bar_area_h + 16, metric, 11, 700, COLORS["ink"], "middle"))


def build_svg() -> str:
    W, H = 900, 340
    pw = 420
    ph = 280

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
        t(W / 2, 28, "NLG Metrics: Full Fine-Tune vs. LoRA (E2E Dataset)", 18, 850, anchor="middle"),
    ]

    small_data = [(m, sf, sl) for m, sf, sl, _, _ in METRICS]
    medium_data = [(m, mf, ml) for m, _, _, mf, ml in METRICS]

    draw_model_panel(parts, 20, 46, pw, ph, "GPT-2 Small", small_data)
    draw_model_panel(parts, 460, 46, pw, ph, "GPT-2 Medium", medium_data)

    # Legend
    lx = W - 200
    ly = 48
    parts.append(f'<rect x="{lx}" y="{ly - 8}" width="12" height="12" rx="2" fill="{COLORS["fft"]}"/>')
    parts.append(t(lx + 16, ly + 3, "FFT", 11, 600))
    parts.append(f'<rect x="{lx + 50}" y="{ly - 8}" width="12" height="12" rx="2" fill="{COLORS["lora"]}"/>')
    parts.append(t(lx + 66, ly + 3, "LoRA", 11, 600))

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/nlg_metrics_comparison.svg"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
