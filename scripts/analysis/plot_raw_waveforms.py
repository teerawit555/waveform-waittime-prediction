from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LONG_COLUMNS = ["wave_id", "sample", "time_ms", "value"]


def read_raw_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input file type: {suffix}. Use .csv or .xlsx.")


def normalize_long_table(df: pd.DataFrame, dt_ms: float) -> pd.DataFrame | None:
    if not {"wave_id", "sample", "value"}.issubset(df.columns):
        return None

    long_df = df.copy()
    if "time_ms" not in long_df.columns:
        long_df["time_ms"] = pd.to_numeric(long_df["sample"], errors="raise") * dt_ms

    ordered_cols = LONG_COLUMNS + [col for col in long_df.columns if col not in LONG_COLUMNS]
    long_df = long_df[ordered_cols].copy()
    long_df["sample"] = pd.to_numeric(long_df["sample"], errors="raise").astype(int)
    long_df["time_ms"] = pd.to_numeric(long_df["time_ms"], errors="raise")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="raise")
    return long_df.sort_values(["wave_id", "sample"]).reset_index(drop=True)


def signal_wide_to_long(df: pd.DataFrame, sample_col: str, dt_ms: float) -> pd.DataFrame | None:
    columns = list(df.columns)
    fallback_sample_col = sample_col if sample_col in df.columns else None
    value_cols = [col for col in columns if str(col).strip().endswith(":")]
    if not value_cols:
        return None

    rows = []
    for index, col in enumerate(value_cols, start=1):
        col_pos = columns.index(col)
        paired_sample_col = columns[col_pos - 1] if col_pos > 0 else None
        if paired_sample_col is None or str(paired_sample_col).strip().endswith(":"):
            paired_sample_col = fallback_sample_col
        if paired_sample_col is None:
            return None

        wave_id = str(col).rstrip(":")
        samples = pd.to_numeric(df[paired_sample_col], errors="raise").astype(int).to_numpy()
        values = pd.to_numeric(df[col], errors="raise").to_numpy(float)
        rows.append(
            pd.DataFrame(
                {
                    "wave_id": wave_id,
                    "sample": samples,
                    "time_ms": np.round(samples * dt_ms, 6),
                    "value": values,
                }
            )
        )

    return pd.concat(rows, ignore_index=True).sort_values(["wave_id", "sample"]).reset_index(drop=True)


def load_waveforms(path: Path, sample_col: str, dt_ms: float) -> pd.DataFrame:
    df = read_raw_table(path)
    long_df = normalize_long_table(df, dt_ms=dt_ms)
    if long_df is None:
        long_df = signal_wide_to_long(df, sample_col=sample_col, dt_ms=dt_ms)
    if long_df is None:
        raise KeyError(
            "Raw file must be long format with wave_id/sample/value columns "
            "or wide format with value columns ending in ':'."
        )
    return long_df


def choose_wave_ids(raw: pd.DataFrame, wave_id: str | None, mode: str, topk: int, seed: int) -> list:
    if wave_id:
        return [wave_id]

    wave_ids = raw["wave_id"].drop_duplicates().tolist()
    limit = len(wave_ids) if topk <= 0 else min(topk, len(wave_ids))
    if mode == "random":
        return pd.Series(wave_ids).sample(n=limit, random_state=seed).tolist()
    return wave_ids[:limit]


def chunked(values: list, size: int) -> list[list]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def safe_file_stem(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return safe or "waveform"


def get_label_ms(wave: pd.DataFrame, label_col: str) -> float | None:
    if label_col not in wave.columns:
        return None

    labels = pd.to_numeric(wave[label_col], errors="coerce").dropna()
    if labels.empty:
        return None
    return float(labels.iloc[0])


def get_wave_title(wave: pd.DataFrame, wave_id: object) -> str:
    title = f"wave_id={wave_id}"
    if "type" not in wave.columns:
        return title
    types = wave["type"].dropna().astype(str)
    if types.empty or not types.iloc[0]:
        return title
    return f"{title} | {types.iloc[0]}"


def add_label_line(ax, label_ms: float | None, show_legend: bool = True) -> None:
    if label_ms is None:
        return
    ax.axvline(
        label_ms,
        color="#F59E0B",
        linestyle="--",
        linewidth=1.8,
        label=f"label = {label_ms:.4f} ms" if show_legend else None,
    )

    x_min, x_max = ax.get_xlim()
    place_on_left = label_ms > (x_min + x_max) / 2
    ax.annotate(
        f"label = {label_ms:.3f} ms",
        xy=(label_ms, 0.98),
        xycoords=("data", "axes fraction"),
        xytext=(-4 if place_on_left else 4, 0),
        textcoords="offset points",
        rotation=90,
        rotation_mode="anchor",
        ha="right",
        va="center",
        fontsize=8,
        color="#B45309",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def plot_separate_waveform_files(
    raw: pd.DataFrame,
    wave_ids: list,
    outdir: Path,
    dpi: int,
    label_col: str,
    show_label: bool,
) -> int:
    count = 0
    for wave_id in wave_ids:
        wave_id_str = str(wave_id)
        g = raw[raw["wave_id"].astype(str) == wave_id_str].copy()
        if len(g) == 0:
            print(f"  [skip] {wave_id_str} - not found in raw file")
            continue

        t = g["time_ms"].to_numpy(dtype=float)
        x = g["value"].to_numpy(dtype=float)
        label_ms = get_label_ms(g, label_col) if show_label else None

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(t, x, linewidth=1.5, color="#00528A", label="waveform")
        add_label_line(ax, label_ms)
        ax.set_xlabel("time_ms")
        ax.set_ylabel("value")
        ax.set_title(get_wave_title(g, wave_id_str))
        ax.grid(True, alpha=0.28)
        ax.legend()
        fig.tight_layout()

        save_path = outdir / f"wave_{safe_file_stem(wave_id_str)}.png"
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
        count += 1

    return count


def plot_waveform_pages(
    raw: pd.DataFrame,
    wave_ids: list,
    outdir: Path,
    per_page: int,
    cols: int,
    dpi: int,
    label_col: str,
    show_label: bool,
) -> int:
    if not wave_ids:
        return 0

    page_size = len(wave_ids) if per_page <= 0 else max(1, per_page)
    col_count = max(1, cols)
    pages = chunked(wave_ids, page_size)

    for page_index, page_wave_ids in enumerate(pages, start=1):
        row_count = math.ceil(len(page_wave_ids) / col_count)
        fig_width = 9.5 * col_count
        fig_height = max(3.0 * row_count, 4.0)
        fig, axes = plt.subplots(row_count, col_count, figsize=(fig_width, fig_height), squeeze=False)
        flat_axes = axes.ravel()

        for ax, wave_id in zip(flat_axes, page_wave_ids):
            wave_id_str = str(wave_id)
            g = raw[raw["wave_id"].astype(str) == wave_id_str].copy()
            if len(g) == 0:
                ax.set_visible(False)
                print(f"  [skip] {wave_id_str} - not found in raw file")
                continue

            t = g["time_ms"].to_numpy(dtype=float)
            x = g["value"].to_numpy(dtype=float)
            label_ms = get_label_ms(g, label_col) if show_label else None
            ax.plot(t, x, linewidth=1.2, color="#00528A")
            add_label_line(ax, label_ms, show_legend=False)
            ax.set_title(get_wave_title(g, wave_id_str), fontsize=10)
            ax.set_xlabel("time_ms")
            ax.set_ylabel("value")
            ax.grid(True, alpha=0.28)

        for ax in flat_axes[len(page_wave_ids) :]:
            ax.set_visible(False)

        fig.suptitle(f"Raw waveforms page {page_index}/{len(pages)}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = outdir / f"raw_waveforms_page_{page_index:03d}.png"
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)

    return len(pages)


def main() -> None:
    ap = argparse.ArgumentParser("plot_raw_waveforms")
    ap.add_argument("--raw", required=True, help="raw waveform CSV/XLSX")
    ap.add_argument("--outdir", required=True, help="output folder for plots")
    ap.add_argument("--topk", type=int, default=30, help="number of waveforms to plot; <=0 plots all")
    ap.add_argument("--layout", choices=["separate", "pages", "both"], default="both", help="save separate images, grouped pages, or both")
    ap.add_argument("--per-page", type=int, default=11, help="number of waveforms per output image; <=0 puts all on one page")
    ap.add_argument("--cols", type=int, default=1, help="number of subplot columns per page")
    ap.add_argument("--dpi", type=int, default=180, help="output image DPI")
    ap.add_argument("--mode", choices=["first", "random"], default="first")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wave-id", default=None, dest="wave_id", help="plot only this single wave_id")
    ap.add_argument("--dt-ms", type=float, default=0.01, help="sample interval in ms for files without time_ms")
    ap.add_argument("--sample-col", default="Signal", help="sample/time column name for wide files")
    ap.add_argument("--label-col", default="wait_time_ms", help="column to draw as vertical label line")
    ap.add_argument("--no-label", action="store_true", help="do not draw label lines")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_waveforms(Path(args.raw), sample_col=args.sample_col, dt_ms=args.dt_ms)
    chosen = choose_wave_ids(raw, args.wave_id, args.mode, args.topk, args.seed)
    show_label = not args.no_label

    if args.layout in {"separate", "both"}:
        plot_count = plot_separate_waveform_files(
            raw, chosen, outdir, args.dpi, args.label_col, show_label
        )
        print(f"Saved {plot_count} raw waveform plot(s) to: {outdir}")

    if args.layout in {"pages", "both"}:
        page_count = plot_waveform_pages(
            raw, chosen, outdir, args.per_page, args.cols, args.dpi, args.label_col, show_label
        )
        print(f"Saved {page_count} raw waveform page(s) for {len(chosen)} waveform(s) to: {outdir}")


if __name__ == "__main__":
    main()
