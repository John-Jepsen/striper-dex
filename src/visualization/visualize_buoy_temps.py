#!/usr/bin/env python3
"""
Elite-level visualizations for weekly Monterey morning SST data exported by buoy-temps.py.

The script expects a CSV containing weekly aggregates with the columns:
    iso_year, iso_week, week_start, week_end, days_count, morning_temp_avg_F

Dependencies: pandas, matplotlib, seaborn, numpy

Example:
    python visualize_buoy_temps.py --input monterey_morning_temps_weekly_last150days.csv \
        --outdir figures --show
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

import matplotlib.dates as mdates  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
from matplotlib.figure import Figure  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]


@dataclass
class VisualizationConfig:
    input_csv: Path
    outdir: Path
    show: bool = False
    dpi: int = 140
    palette: str = "crest"
    recent_window_weeks: int = 26
    rolling_window_weeks: int = 4


def load_weekly_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["week_start", "week_end"])
    required_cols = {
        "iso_year",
        "iso_week",
        "week_start",
        "week_end",
        "days_count",
        "morning_temp_avg_F",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Normalised metadata for richer visualizations and future joins.
    df = df.sort_values("week_start").reset_index(drop=True)
    df["week_mid"] = df["week_start"] + pd.Timedelta(days=3)  # type: ignore[operator]
    # Use the ISO calendar processing from pandas for safe week/year handling.
    iso = df["week_start"].dt.isocalendar()
    df["iso_year"] = iso["year"].astype(int)
    df["iso_week"] = iso["week"].astype(int)
    df["iso_year_week"] = (
        df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)
    )
    df["week_of_year"] = df["iso_week"]
    df["year"] = df["iso_year"]
    df["week_index"] = np.arange(len(df))

    # Baseline stats for anomaly visualizations.
    df["climatology"] = df.groupby("week_of_year")["morning_temp_avg_F"].transform(
        "median"
    )
    df["anomaly"] = df["morning_temp_avg_F"] - df["climatology"]
    return df


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_or_show(fig: Figure, cfg: VisualizationConfig, filename: str) -> None:
    output_path = cfg.outdir / filename
    fig.savefig(str(output_path), dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show:
        fig.show()
    plt.close(fig)


def plot_timeseries(df: pd.DataFrame, cfg: VisualizationConfig) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False})

    ax.plot(
        df["week_mid"],
        df["morning_temp_avg_F"],
        color="#0b7285",
        linewidth=1.6,
        label="Weekly average",
    )

    rolling = df["morning_temp_avg_F"].rolling(cfg.rolling_window_weeks, min_periods=1).mean()
    ax.plot(df["week_mid"], rolling, color="#dc2f02", linewidth=2.3, label=f"{cfg.rolling_window_weeks}-week mean")

    recent_df = df.tail(cfg.recent_window_weeks)
    ax.scatter(
        recent_df["week_mid"],
        recent_df["morning_temp_avg_F"],
        s=32,
        color="#6a994e",
        label=f"Last {cfg.recent_window_weeks} weeks",
        zorder=5,
    )

    min_idx = int(df["morning_temp_avg_F"].idxmin())
    max_idx = int(df["morning_temp_avg_F"].idxmax())
    for idx, label in ((min_idx, "Min"), (max_idx, "Max")):
        temp = cast(float, df.at[idx, "morning_temp_avg_F"])
        week_mid = cast(pd.Timestamp, df.at[idx, "week_mid"])
        iso_week = cast(int, df.at[idx, "iso_week"])
        iso_year = cast(int, df.at[idx, "iso_year"])
        ax.annotate(
            f"{label}: {temp:.2f}°F\nweek {iso_week:02d}, {iso_year}",
            xy=(week_mid, temp),  # type: ignore[arg-type]
            xytext=(15, 15) if label == "Max" else (-15, -25),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#495057"),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#adb5bd", alpha=0.9),
        )

    ax.set_title("Weekly Monterey Morning SST", fontsize=16, weight="semibold")
    ax.set_ylabel("Temperature (°F)")
    ax.set_xlabel("Week midpoint")
    ax.legend(frameon=False, loc="upper left")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    return fig


def plot_anomaly_bars(df: pd.DataFrame, cfg: VisualizationConfig) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False})

    colors = np.where(df["anomaly"] >= 0, "#ef233c", "#023047")
    bars = ax.bar(
        df["week_mid"],
        df["anomaly"],
        width=5,
        color=colors,
        edgecolor="none",
    )

    # Add a thin zero line to emphasise anomaly sign changes.
    ax.axhline(0, color="#6c757d", linewidth=0.9)

    ax.set_title("Weekly SST Anomalies vs median climatology", fontsize=15, weight="semibold")
    ax.set_ylabel("Anomaly (°F)")
    ax.set_xlabel("ISO week")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    # Annotate peak positive/negative anomalies only if dataset is large enough to matter.
    for idx in (int(df["anomaly"].idxmax()), int(df["anomaly"].idxmin())):
        anomaly = cast(float, df.at[idx, "anomaly"])
        if np.isfinite(anomaly):
            iso_week = cast(int, df.at[idx, "iso_week"])
            iso_year = cast(int, df.at[idx, "iso_year"])
            week_mid = cast(pd.Timestamp, df.at[idx, "week_mid"])
            ax.annotate(
                f"{anomaly:+.2f}°F\nweek {iso_week:02d}, {iso_year}",
                xy=(week_mid, anomaly),  # type: ignore[arg-type]
                xytext=(0, 28 if anomaly >= 0 else -32),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#adb5bd", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="#adb5bd"),
            )

    return fig


def plot_seasonal_heatmap(df: pd.DataFrame, cfg: VisualizationConfig) -> Figure:
    pivot = df.pivot_table(
        index="iso_week",
        columns="iso_year",
        values="morning_temp_avg_F",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="white")
    sns.heatmap(
        pivot,
        cmap=cfg.palette,
        ax=ax,
        cbar_kws={"label": "Temperature (°F)"},
        linewidths=0.4,
        linecolor="white",
        annot=len(pivot.columns) <= 5,
        fmt=".1f",
        annot_kws={"fontsize": 8},
    )

    ax.set_title("Seasonal profile by ISO week", fontsize=15, weight="semibold")
    ax.set_xlabel("ISO year")
    ax.set_ylabel("ISO week")
    ax.invert_yaxis()

    return fig


def render_all(df: pd.DataFrame, cfg: VisualizationConfig) -> Iterable[str]:
    outputs = []

    fig = plot_timeseries(df, cfg)
    filename = "monterey_weekly_timeseries.png"
    save_or_show(fig, cfg, filename)
    outputs.append(filename)

    fig = plot_anomaly_bars(df, cfg)
    filename = "monterey_weekly_anomalies.png"
    save_or_show(fig, cfg, filename)
    outputs.append(filename)

    fig = plot_seasonal_heatmap(df, cfg)
    filename = "monterey_weekly_heatmap.png"
    save_or_show(fig, cfg, filename)
    outputs.append(filename)

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create elite-level SST visualizations.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("monterey_morning_temps_weekly_last150days.csv"),
        help="Weekly CSV produced by buoy-temps.py",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("figures"),
        help="Where to save generated figures",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Figure resolution in dots per inch",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="crest",
        help="Seaborn palette name for heatmaps",
    )
    parser.add_argument(
        "--recent-window-weeks",
        type=int,
        default=26,
        help="Highlight the most recent N weeks in the time series plot",
    )
    parser.add_argument(
        "--rolling-window-weeks",
        type=int,
        default=4,
        help="Window length for the trend line rolling mean",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(args=args)

    cfg = VisualizationConfig(
        input_csv=ns.input,
        outdir=ns.outdir,
        show=ns.show,
        dpi=ns.dpi,
        palette=ns.palette,
        recent_window_weeks=ns.recent_window_weeks,
        rolling_window_weeks=ns.rolling_window_weeks,
    )

    ensure_outdir(cfg.outdir)
    df = load_weekly_csv(cfg.input_csv)

    outputs = render_all(df, cfg)
    print("Generated figures:")
    for name in outputs:
        print(f"  {cfg.outdir / name}")


if __name__ == "__main__":
    main()
