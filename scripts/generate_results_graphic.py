#!/usr/bin/env python3
"""Generate simple poster-ready LoRA vs full fine-tuning result graphics.

The output is intentionally direct: three separate panels compare LoRA against
full fine-tuning for trainable parameters, peak VRAM, and epoch time. In every
panel, full fine-tuning is the 100% baseline and the teal bar shows LoRA's cost
as a percent of that baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    name: str
    fft_params: float
    lora_params: float
    fft_vram_mb: float
    lora_vram_mb: float
    fft_epoch_s: float
    lora_epoch_s: float


EXPERIMENTS = [
    Experiment(
        name="SST-2 / RoBERTa",
        fft_params=124_647_170,
        lora_params=887_042,
        fft_vram_mb=2483.524,
        lora_vram_mb=1317.474,
        fft_epoch_s=205.01,
        lora_epoch_s=177.86,
    ),
    Experiment(
        name="MNLI / RoBERTa",
        fft_params=124_647_170,
        lora_params=887_811,
        fft_vram_mb=14_708.07,
        lora_vram_mb=1_317.48,
        fft_epoch_s=5_098.02,
        lora_epoch_s=1_003.95,
    ),
    Experiment(
        name="CIFAR-10 / ViT",
        fft_params=85_806_346,
        lora_params=302_602,
        fft_vram_mb=3522.9,
        lora_vram_mb=2296.7,
        fft_epoch_s=485.5,
        lora_epoch_s=374.3,
    ),
    Experiment(
        name="E2E / GPT-2 S",
        fft_params=124_439_808,
        lora_params=147_456,
        fft_vram_mb=7751.6,
        lora_vram_mb=3624.37,
        fft_epoch_s=1141.8,
        lora_epoch_s=99.78,
    ),
    Experiment(
        name="E2E / GPT-2 M",
        fft_params=124_439_808,
        lora_params=393_216,
        fft_vram_mb=7134.2,
        lora_vram_mb=3624.37,
        fft_epoch_s=7908.0,
        lora_epoch_s=5088.0,
    ),
]


COLORS = {
    "bg": "#ffffff",
    "panel": "#ffffff",
    "ink": "#172026",
    "muted": "#65737e",
    "grid": "#d9dfdc",
    "fft": "#d5d8dc",
    "lora": "#0f766e",
    "lora_dark": "#0b514c",
    "accent": "#f59e0b",
}


def fmt_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def fmt_time(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def fmt_mb(value: float) -> str:
    if value >= 1024:
        return f"{value / 1024:.1f}GB"
    return f"{value:.0f}MB"


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


def metric_specs() -> list[dict]:
    return [
        {
            "title": "Trainable Parameters",
            "getter": lambda e: (e.fft_params, e.lora_params),
            "unit": fmt_count,
            "lower_label": "fewer trainable params",
        },
        {
            "title": "Peak VRAM",
            "getter": lambda e: (e.fft_vram_mb, e.lora_vram_mb),
            "unit": fmt_mb,
            "lower_label": "less VRAM",
        },
        {
            "title": "Average Epoch Time",
            "getter": lambda e: (e.fft_epoch_s, e.lora_epoch_s),
            "unit": fmt_time,
            "lower_label": "faster epochs",
        },
    ]


def draw_panel(x: float, y: float, w: float, h: float, spec: dict) -> list[str]:
    parts: list[str] = []
    parts.append(rect(x, y, w, h, COLORS["panel"], 10, "#d8ded9"))
    parts.append(svg_text(x + 32, y + 58, spec["title"], 44, 800))

    label_w = 250
    bar_x = x + label_w + 48
    bar_w = w - label_w - 110
    row_y = y + 126
    row_gap = 76
    bar_h = 28

    # Axis hints: full fine-tuning is always the full-width 100% baseline.
    parts.append(svg_text(bar_x, y + 100, "0%", 24, 700, COLORS["muted"], "middle"))
    parts.append(svg_text(bar_x + bar_w, y + 100, "FFT = 100%", 24, 700, COLORS["muted"], "middle"))
    parts.append(
        f'<line x1="{bar_x:.1f}" y1="{y + 106:.1f}" x2="{bar_x + bar_w:.1f}" '
        f'y2="{y + 106:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>'
    )

    for idx, exp in enumerate(EXPERIMENTS):
        fft_value, lora_value = spec["getter"](exp)
        lora_pct = 100.0 * lora_value / fft_value
        reduction = 100.0 - lora_pct
        cy = row_y + idx * row_gap
        lora_w = max(3.0, bar_w * min(lora_pct, 100.0) / 100.0)

        parts.append(svg_text(x + 32, cy + 25, exp.name, 28, 750))
        parts.append(rect(bar_x, cy, bar_w, bar_h, COLORS["fft"], 14))
        parts.append(rect(bar_x, cy, lora_w, bar_h, COLORS["lora"], 14))
        parts.append(
            svg_text(
                bar_x,
                cy + 62,
                f"FFT {spec['unit'](fft_value)}",
                24,
                600,
                COLORS["muted"],
            )
        )
        parts.append(
            svg_text(
                bar_x + 218,
                cy + 62,
                f"LoRA {spec['unit'](lora_value)} ({lora_pct:.1f}% of FFT)",
                24,
                700,
                COLORS["lora_dark"],
            )
        )

    return parts


def build_svg() -> str:
    width, height = 1800, 1930
    panel_x, panel_w, panel_h = 80, 1640, 535
    panel_gap = 56

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, COLORS["bg"]),
        svg_text(80, 88, "LoRA vs. Full Fine-Tuning Results", 62, 850),
        svg_text(
            80,
            140,
            "Gray bars are full fine-tuning baselines. Teal bars are LoRA for the same model/task run.",
            34,
            400,
            COLORS["muted"],
        ),
    ]

    start_y = 190
    for idx, spec in enumerate(metric_specs()):
        parts.extend(draw_panel(panel_x, start_y + idx * (panel_h + panel_gap), panel_w, panel_h, spec))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lora_fft_cost_comparison.svg"),
        help="Path to write the generated SVG.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
