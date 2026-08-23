"""
Assemble the ablation table (EXPERIMENTS.md Table 1) from one or more results.json
files written by run_eval.py, and print the headline sentence.

Each run_eval results.json holds BOTH a "baseline" (static SAHI) block and an
"adaptive" block for that run's config. So:
  * the STATIC row (A) is taken from the `baseline` block of the first file
    (every run re-computes it; it's the cross-check), and
  * every named row contributes its `adaptive` block.

Usage:
  python eval/make_table.py \
      A=results/A.json B=results/B.json C=results/C.json \
      D=results/D.json E=results/E.json F=results/F_full.json \
      --out results/table1.md

  # or just point it at the full run for a 2-row static-vs-full summary:
  python eval/make_table.py F=results/F_full.json

Pure stdlib — no numpy needed.
"""

import argparse
import json
import os


COLS = ["mAP@50", "mAP@50:95", "AP_small", "FP/frame", "P@R0.8", "tiles"]


def _metrics(block):
    """Pull the six table columns out of an evaluate() report block."""
    pr_key = next(k for k in block if k.startswith("precision@recall"))
    comp = block.get("compute", {}) or {}
    return {
        "mAP@50": block["mAP_50"],
        "mAP@50:95": block["mAP_50_95"],
        "AP_small": block["AP_small_50_95"],
        "FP/frame": block["clutter@op_conf"]["fp_per_frame"],
        "P@R0.8": block[pr_key]["precision"],
        "tiles": comp.get("mean_tiles_processed", 0.0),
        "_reduction": comp.get("mean_tile_reduction", 0.0),
    }


def _fmt_row(label, m):
    cells = " | ".join(f"{m[c]:.3f}" if c != "tiles" else f"{m[c]:.1f}" for c in COLS)
    return f"| {label:14s} | {cells} |"


def build(rows, static_from=None):
    """rows: list of (label, path). Returns (markdown_table, headline)."""
    lines = [f"| {'Config':14s} | " + " | ".join(COLS) + " |",
             "|" + "-" * 16 + "|" + "|".join(["-" * (len(c) + 2) for c in COLS]) + "|"]

    # static baseline row from the first (or designated) file's `baseline` block
    src = static_from or rows[0][1]
    static = _metrics(json.load(open(src))["baseline"])
    lines.append(_fmt_row("A static", static))

    full = None
    for label, path in rows:
        m = _metrics(json.load(open(path))["adaptive"])
        lines.append(_fmt_row(label, m))
        if label.lower().startswith(("f", "full")) or "full" in label.lower():
            full = m
    full = full or m  # default to the last row if none looks like 'full'

    # headline sentence (EXPERIMENTS.md §1)
    def pct(a, b):
        return (a - b) / (a + 1e-9) * 100
    tile_pct = pct(static["tiles"], full["tiles"])
    tile_word = "fewer" if tile_pct >= 0 else "more"
    headline = (
        f"Full adaptive tiling cut FP/frame from {static['FP/frame']:.2f} to "
        f"{full['FP/frame']:.2f} ({pct(static['FP/frame'], full['FP/frame']):+.0f}%) "
        f"and raised P@R0.8 from {static['P@R0.8']:.3f} to {full['P@R0.8']:.3f}, "
        f"at {abs(tile_pct):.0f}% {tile_word} tiles, with "
        f"ΔmAP = {full['mAP@50:95'] - static['mAP@50:95']:+.3f}."
    )
    return "\n".join(lines), headline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("rows", nargs="+", metavar="LABEL=results.json",
                   help="one or more adaptive-config rows")
    p.add_argument("--static-from", default=None,
                   help="results.json to take the static (A) row from "
                        "(default: the first row's file)")
    p.add_argument("--out", default=None, help="write the markdown table here too")
    return p.parse_args()


def main():
    a = parse_args()
    rows = []
    for item in a.rows:
        if "=" not in item:
            raise SystemExit(f"expected LABEL=path, got {item!r}")
        label, path = item.split("=", 1)
        rows.append((label, path))

    table, headline = build(rows, static_from=a.static_from)
    print(table)
    print("\n" + headline)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write("# Table 1 — adaptive tiling ablation\n\n")
            f.write(table + "\n\n> " + headline + "\n")
        print(f"\n[make_table] wrote {a.out}")


if __name__ == "__main__":
    main()
