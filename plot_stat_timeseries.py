"""
Plot SAL vs BP stats from stat_timeseries.json (epoch summary row + intra-epoch grid).

Usage:
  python plot_stat_timeseries.py --exp_root runs --run_name sal_cmp
  python plot_stat_timeseries.py --exp_root runs --run_name sal_cmp --output my_fig.png
  python plot_stat_timeseries.py --exp_root runs --run_name sal_cmp --core_only --output alignment.pdf
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

# Consistent train / test styling across epoch summary panels
TRAIN_COLOR = "#1f77b4"
TEST_COLOR = "#ff7f0e"
TRAIN_MARKER = "o"
TEST_MARKER = "s"

# Epoch summary row layout (tune here).
EPOCH_XLABEL_PAD = 0  # points between x ticks and "Epoch"; lower = label moves up
EPOCH_INTRA_HSPACE = 0.18  # vertical gap between epoch row and first intra-epoch row


def resolve_run_dir(exp_root: str, run_name: str) -> str:
    return os.path.normpath(os.path.join(os.path.expanduser(exp_root), run_name))


def _normalize_epoch_metrics(em):
    """Unify old/new JSON field names for Net1-only epoch summary."""
    if not em:
        return None
    out = dict(em)
    if "test_acc_net1" not in out and out.get("test_acc_net1_sal"):
        out["test_acc_net1"] = out["test_acc_net1_sal"]
    if "test_loss_ff" not in out:
        tl = out.get("test_loss") or out.get("test_losses")
        if isinstance(tl, dict):
            out["test_loss_ff"] = tl.get("feedforward", [])
        elif isinstance(tl, list):
            out["test_loss_ff"] = tl
    if "train_acc_net1" not in out and out.get("train_acc_net1_sal"):
        out["train_acc_net1"] = out["train_acc_net1_sal"]
    return out


def _infer_train_acc_from_series(ts) -> tuple[list[int], list[float]] | None:
    """Mean intra-epoch acc_net1 per epoch (fallback for older runs)."""
    if not ts.get("acc_net1") or not ts.get("epoch"):
        return None
    by_ep = defaultdict(list)
    for e, a in zip(ts["epoch"], ts["acc_net1"]):
        if a is not None and not (isinstance(a, float) and np.isnan(a)):
            by_ep[int(e)].append(float(a))
    if not by_ep:
        return None
    ep_sorted = sorted(by_ep)
    return ep_sorted, [float(np.mean(by_ep[e])) for e in ep_sorted]


def _epoch_metrics_from_summary(run_dir: str):
    summary_path = os.path.join(run_dir, "sal_bp_compare_run.json")
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, "r") as f:
        summary = json.load(f)
    if summary.get("epoch_metrics"):
        return _normalize_epoch_metrics(summary["epoch_metrics"])
    losses = summary.get("losses") or {}
    acc = summary.get("accuracies") or {}
    test_l = summary.get("test_losses") or {}
    ff = losses.get("feedforward")
    gn = losses.get("gradient")
    if not ff or not gn:
        return None
    n = len(ff)
    em = {
        "epoch": list(range(1, n + 1)),
        "train_loss_ff": ff,
        "train_loss_gn": gn,
        "test_acc_net1": acc.get("net1_sal", acc.get("net1", [])),
    }
    if test_l.get("feedforward"):
        em["test_loss_ff"] = test_l["feedforward"]
    train_acc = acc.get("net1_sal_train") or acc.get("net1_train")
    if train_acc:
        em["train_acc_net1"] = train_acc
    return _normalize_epoch_metrics(em)


def load_timeseries(path: str, run_dir: str | None = None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stat file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if "series" in data:
        ts = data["series"]
        num_layers = data.get("num_layers")
    else:
        ts = data
        num_layers = data.get("num_layers")
    epoch_metrics = _normalize_epoch_metrics(data.get("epoch_metrics"))
    if epoch_metrics is None and run_dir:
        epoch_metrics = _epoch_metrics_from_summary(run_dir)
    if epoch_metrics is not None and not epoch_metrics.get("train_acc_net1"):
        inferred = _infer_train_acc_from_series(ts)
        if inferred is not None:
            ep_inf, acc_inf = inferred
            epoch_metrics = dict(epoch_metrics)
            if not epoch_metrics.get("epoch"):
                epoch_metrics["epoch"] = ep_inf
            elif list(epoch_metrics["epoch"]) == ep_inf:
                epoch_metrics["train_acc_net1"] = acc_inf
            else:
                acc_by_ep = dict(zip(ep_inf, acc_inf))
                epoch_metrics["train_acc_net1"] = [
                    acc_by_ep.get(int(e), float("nan")) for e in epoch_metrics["epoch"]
                ]
    if num_layers is None and ts.get("w_cos_layer"):
        num_layers = len(ts["w_cos_layer"][0])
    if num_layers is None:
        raise ValueError(
            "Could not infer num_layers; re-run training with an updated "
            "stat_timeseries.json or pass --num_layers."
        )
    return ts, int(num_layers), data, epoch_metrics


def _apply_pub_style(dpi=150):
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": dpi,
            "savefig.dpi": max(dpi, 200),
            # Embed TrueType fonts; keep PDF/PS as vectors (not Type-3 bitmaps).
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "axes.labelpad": 10,
            "ytick.major.pad": 6,
            "grid.alpha": 0.28,
            "lines.linewidth": 1.4,
            "axes.grid": True,
        }
    )


def _style_axis(ax):
    ax.grid(True, alpha=0.28, linewidth=0.6)
    ax.tick_params(direction="out", top=False, right=False, pad=4)


def _style_epoch_axis(ax):
    """Tighter padding for the compact epoch summary row."""
    _style_axis(ax)
    ax.tick_params(pad=2)
    ax.xaxis.labelpad = EPOCH_XLABEL_PAD
    ax.yaxis.labelpad = 4


def _set_centered_suptitle(fig, text, fontsize=11, y=0.992):
    """Center title over the horizontal span of all axes (not the full figure)."""
    axes = fig.get_axes()
    if axes:
        x0 = min(ax.get_position().x0 for ax in axes)
        x1 = max(ax.get_position().x1 for ax in axes)
        cx = 0.5 * (x0 + x1)
    else:
        cx = 0.5
    fig.suptitle(text, fontsize=fontsize, x=cx, ha="center", y=y)


def _finite_values(arr):
    a = np.asarray(arr, dtype=float).ravel()
    return a[np.isfinite(a)]


def _series_has_data(ts, gkey, lkey=None, n_layer_cols=0):
    if gkey not in ts or not ts[gkey]:
        return False
    if len(_finite_values(ts[gkey])) > 0:
        return True
    if not lkey or lkey not in ts or not ts[lkey]:
        return False
    for li in range(n_layer_cols):
        vals = []
        for rec in ts[lkey]:
            if li < len(rec) and rec[li] is not None:
                vals.append(rec[li])
        if len(_finite_values(vals)) > 0:
            return True
    return False


# Paper notation: junction pairs compare V_{i+1} with transposed V_bar_i.
_PAPER_VBAR_PAIR = r"\bar{V}_i^{\mathrm{T}}"


def _layer_tick_label(li, is_pair, gkey="", lkey=""):
    """Subplot column title: W/BP rows use layer_i; V / V_bar rows use bridge notation."""
    if is_pair:
        return rf"$V_{{{li+1}}}$–$\bar{{V}}_{{{li}}}^{{\mathrm{{T}}}}$"
    if lkey in (
        "w_cos_layer",
        "w_angle_layer",
        "ep_pathway_norm_layer",
        "ep_pathway_rel_layer",
        "epg_pathway_norm_layer",
        "epg_pathway_rel_layer",
        "epg_pathway_cos_layer",
        "epg_pathway_norm_div_g_layer",
        "epg_bar_pathway_norm_layer",
        "epg_bar_pathway_rel_layer",
        "epg_bar_pathway_cos_layer",
        "epg_bar_pathway_norm_div_bar_h_layer",
    ):
        return f"layer {li}"
    if "bar" in gkey or (lkey and "bar" in lkey):
        return rf"$\bar{{V}}_{{{li}}}$"
    return rf"$V_{{{li}}}$"


def _na_note_for_layer(gkey, li, num_layers):
    """Short note when a grid column is empty by SAL architecture, not a recording bug."""
    if gkey.startswith("v_update_") and "bar" not in gkey and li == 0:
        return "no SAL grad on $V_0$"
    if gkey.startswith("v_grad_abs") and "bar" not in gkey and li == 0:
        return "no SAL grad on $V_0$"
    if gkey.startswith("v_bar_") and li == num_layers - 1:
        return "no $\\bar{V}$ at last layer"
    return None


def _row_plot_opts(gkey):
    """Per-metric axis limits (linear scale only)."""
    if gkey in ("w_cos_global", "w_cos_layer", "v_paired_weight_cos_global", "v_paired_weight_cos_layer"):
        return {"ylim": (0.0, 1.0)}
    if gkey in (
        "epg_pathway_cos_global",
        "epg_pathway_cos_layer",
        "epg_bar_pathway_cos_global",
        "epg_bar_pathway_cos_layer",
    ):
        return {"ylim": (-1.0, 1.0)}
    if gkey in ("w_angle_global", "w_angle_layer"):
        return {"ylim": (20.0, 90.0)}
    return {"ylim": None}


def _column_plot_opts(gkey, layer_idx, n_layers):
    """Per-layer column overrides (last layer: cos→1, angle→0 need wider y windows)."""
    opts = dict(_row_plot_opts(gkey))
    if layer_idx != n_layers - 1:
        return opts
    if gkey in ("w_cos_global", "w_cos_layer"):
        opts["ylim"] = (0.0, 1.1)
    elif gkey in ("w_angle_global", "w_angle_layer"):
        opts["ylim"] = (-10.0, 90.0)
    return opts


def _intra_row_catalog(num_layers):
    n_pair = max(0, num_layers - 1)
    vbar = _PAPER_VBAR_PAIR
    return [
        ("w_cos_global", "w_cos_layer", "Cos. sim. w/ BP", num_layers, False),
        ("w_angle_global", "w_angle_layer", "Angle w/ BP", num_layers, False),
        ("ep_pathway_norm_global", "ep_pathway_norm_layer", r"$\|E_p\|$", num_layers, False),
        (
            "ep_pathway_rel_global",
            "ep_pathway_rel_layer",
            r"$\|E_p\|/\|F\|$",
            num_layers,
            False,
        ),
        ("epg_pathway_norm_global", "epg_pathway_norm_layer", r"$\|E_{pg}\|$", num_layers, False),
        (
            "epg_pathway_norm_div_g_global",
            "epg_pathway_norm_div_g_layer",
            r"$\|E_{pg}\|/\|g\|$",
            num_layers,
            False,
        ),
        (
            "epg_pathway_rel_global",
            "epg_pathway_rel_layer",
            r"$\|E_{pg}\|/\|Fg\|$",
            num_layers,
            False,
        ),
        (
            "epg_pathway_cos_global",
            "epg_pathway_cos_layer",
            r"$\cos(Fg,Bg)$",
            num_layers,
            False,
        ),
        (
            "epg_bar_pathway_norm_global",
            "epg_bar_pathway_norm_layer",
            r"$\|\bar E_{pg}\|$",
            num_layers,
            False,
        ),
        (
            "epg_bar_pathway_norm_div_bar_h_global",
            "epg_bar_pathway_norm_div_bar_h_layer",
            r"$\|\bar E_{pg}\|/\|\bar h\|$",
            num_layers,
            False,
        ),
        (
            "epg_bar_pathway_rel_global",
            "epg_bar_pathway_rel_layer",
            r"$\|\bar E_{pg}\|/\|\bar T\bar h\|$",
            num_layers,
            False,
        ),
        (
            "epg_bar_pathway_cos_global",
            "epg_bar_pathway_cos_layer",
            r"$\cos(\bar T\bar h,\bar h)$",
            num_layers,
            False,
        ),
        ("v_grad_abs_global", "v_grad_abs_layer", r"$|\nabla V|$", num_layers, False),
        ("v_bar_grad_abs_global", "v_bar_grad_abs_layer", r"$|\nabla \bar{V}|$", num_layers, False),
        (
            "v_cross_weight_diff_global",
            "v_cross_weight_diff_layer",
            rf"$|V_{{i+1}} - {vbar}|$",
            n_pair,
            True,
        ),
        (
            "v_paired_weight_cos_global",
            "v_paired_weight_cos_layer",
            rf"$\cos(V_{{i+1}}, {vbar})$",
            n_pair,
            True,
        ),
        ("v_update_no_wd_global", "v_update_no_wd_layer", r"$|\nabla V|$ (pre-WD)", num_layers, False),
        (
            "v_update_wd_global",
            "v_update_wd_layer",
            r"$|\lambda_{\mathrm{decay}}\, V|$",
            num_layers,
            False,
        ),
        (
            "v_update_total_global",
            "v_update_total_layer",
            r"$|\nabla V + \lambda_{\mathrm{decay}}\, V|$",
            num_layers,
            False,
        ),
        (
            "v_bar_update_no_wd_global",
            "v_bar_update_no_wd_layer",
            r"$|\nabla \bar{V}|$ (pre-WD)",
            num_layers,
            False,
        ),
        (
            "v_bar_update_wd_global",
            "v_bar_update_wd_layer",
            r"$|\lambda_{\mathrm{decay}}\, \bar{V}|$",
            num_layers,
            False,
        ),
        (
            "v_bar_update_total_global",
            "v_bar_update_total_layer",
            r"$|\nabla \bar{V} + \lambda_{\mathrm{decay}}\, \bar{V}|$",
            num_layers,
            False,
        ),
        ("v_weight_abs_global", "v_weight_abs_layer", r"$|V|$", num_layers, False),
        ("v_bar_weight_abs_global", "v_bar_weight_abs_layer", r"$|\bar{V}|$", num_layers, False),
    ]


def _active_intra_rows(ts, num_layers):
    rows = []
    for gkey, lkey, ylabel, n_data_layers, is_pair in _intra_row_catalog(num_layers):
        if _series_has_data(ts, gkey, lkey, n_data_layers):
            rows.append(
                {
                    "gkey": gkey,
                    "lkey": lkey,
                    "ylabel": ylabel,
                    "n_data_layers": n_data_layers,
                    "n_grid_layers": num_layers,
                    "is_pair": is_pair,
                }
            )
    return rows


def _row_letter(row_index: int) -> str:
    """A, B, ..., Z, AA, AB, ... for figure row labels."""
    n = row_index
    letters = ""
    while True:
        letters = chr(ord("A") + (n % 26)) + letters
        n = n // 26 - 1
        if n < 0:
            break
    return letters


# Figure 1 ends at junction cos; figure 2 = traced updates and weight norms.
_SPLIT_AFTER_GKEY = "v_paired_weight_cos_global"

# Compact summary figure: key alignment metrics only (order preserved).
_CORE_ROW_GKEYS = (
    "w_cos_global",
    "ep_pathway_norm_global",
    "ep_pathway_rel_global",
    "v_grad_abs_global",
    "v_bar_grad_abs_global",
    "v_cross_weight_diff_global",
    "v_paired_weight_cos_global",
)


def _select_active_rows_by_gkeys(active_rows, gkeys):
    """Keep only rows whose global key is in *gkeys*, in catalog order."""
    by_gkey = {row["gkey"]: row for row in active_rows}
    return [by_gkey[k] for k in gkeys if k in by_gkey]


def _split_active_rows(active_rows, part: str):
    """``align``: rows through cos(V, V_bar); ``updates``: remaining rows."""
    split_at = None
    for i, row in enumerate(active_rows):
        if row["gkey"] == _SPLIT_AFTER_GKEY:
            split_at = i
            break
    if split_at is None:
        return list(active_rows) if part == "align" else []
    if part == "align":
        return active_rows[: split_at + 1]
    if part == "updates":
        return active_rows[split_at + 1 :]
    raise ValueError(f"part must be 'align' or 'updates', got {part!r}")


def _global_row_letter(row, letter_rows, offset=0):
    """Letter by position among *letter_rows* (the rows drawn in this figure).

    Important: use the plotted subset, not the full metric catalog. Looking up
    indices in the full catalog skips letters for omitted rows (e.g. core plots
    omit ``w_angle`` and previously jumped from B to D).
    """
    for i, r in enumerate(letter_rows):
        if r["gkey"] == row["gkey"]:
            return _row_letter(i + offset)
    return None


def _savefig_pub(fig, out_path, dpi=150, pad_inches=0.02):
    """Save PNG (raster) or PDF/SVG/EPS (vector paths + embedded fonts)."""
    ext = os.path.splitext(out_path)[1].lower()
    kwargs = {"bbox_inches": "tight", "pad_inches": pad_inches}
    if ext in (".pdf", ".svg", ".eps"):
        # Vector backends: omit dpi so line art / text stay resolution-independent.
        fig.savefig(out_path, format=ext.lstrip("."), **kwargs)
    else:
        fig.savefig(out_path, dpi=max(dpi, 200), **kwargs)


def _plot_row_letter_axis(ax, letter):
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        letter,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )


def _plot_epoch_row(
    fig, gs_row, n_data_cols, width_ratios, epoch_metrics, *, label_rows, panel_wspace=0.30
):
    """Loss/acc centered in the data-column span; label column aligned with intra rows."""
    n_cols = len(width_ratios)
    gs_ep = gs_row.subgridspec(1, n_cols, width_ratios=width_ratios, wspace=panel_wspace)
    gs_loss_acc = gs_ep[0, 0:n_data_cols].subgridspec(
        1, 2, width_ratios=[1.0, 1.0], wspace=panel_wspace
    )
    _plot_epoch_loss_panel(fig.add_subplot(gs_loss_acc[0, 0]), epoch_metrics)
    _plot_epoch_acc_panel(fig.add_subplot(gs_loss_acc[0, 1]), epoch_metrics)
    if label_rows:
        _plot_row_letter_axis(fig.add_subplot(gs_ep[0, n_data_cols]), "A")


def build_row_manifest(active_rows, letter_offset=0):
    """Row letter → metric metadata (for LaTeX captions / training_stats_rows.json)."""
    manifest = []
    for row in active_rows:
        manifest.append(
            {
                "row": _global_row_letter(row, active_rows, offset=letter_offset),
                "gkey": row["gkey"],
                "lkey": row["lkey"],
                "ylabel": row["ylabel"],
                "is_pair": row["is_pair"],
                "n_data_layers": row["n_data_layers"],
            }
        )
    return manifest


def save_row_manifest(manifest, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rows": manifest}, f, indent=2)


def _plot_epoch_loss_panel(ax, epoch_metrics):
    ep = np.array(epoch_metrics["epoch"], dtype=float)
    me = max(1, len(ep) // 12)
    ax.plot(
        ep,
        epoch_metrics["train_loss_ff"],
        f"{TRAIN_MARKER}-",
        color=TRAIN_COLOR,
        markersize=4,
        markevery=me,
        label="Train",
    )
    if epoch_metrics.get("test_loss_ff"):
        ax.plot(
            ep,
            epoch_metrics["test_loss_ff"],
            f"{TEST_MARKER}-",
            color=TEST_COLOR,
            markersize=4,
            markevery=me,
            label="Test",
        )
    ax.set_ylabel("Cross-entropy")
    ax.set_xlabel("Epoch")
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="0.8")
    _style_epoch_axis(ax)
    ax.set_title("Loss", pad=3)


def _plot_epoch_acc_panel(ax, epoch_metrics):
    ep = np.array(epoch_metrics["epoch"], dtype=float)
    me = max(1, len(ep) // 12)
    train_acc = epoch_metrics.get("train_acc_net1") or epoch_metrics.get("train_acc_net1_sal")
    test_acc = epoch_metrics.get("test_acc_net1") or epoch_metrics.get("test_acc_net1_sal")
    if train_acc:
        ax.plot(
            ep,
            train_acc,
            f"{TRAIN_MARKER}-",
            color=TRAIN_COLOR,
            markersize=4,
            markevery=me,
            label="Train",
        )
    if test_acc:
        ax.plot(
            ep,
            test_acc,
            f"{TEST_MARKER}-",
            color=TEST_COLOR,
            markersize=4,
            markevery=me,
            label="Test",
        )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Epoch")
    if train_acc or test_acc:
        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="0.8")
    _style_epoch_axis(ax)
    ax.set_title("Accuracy", pad=3)


def _layer_series(ts, lkey, li):
    if not lkey or lkey not in ts or not ts[lkey]:
        return None
    n_rec = len(ts[lkey])
    return np.array(
        [
            ts[lkey][rec_i][li] if li < len(ts[lkey][rec_i]) else np.nan
            for rec_i in range(n_rec)
        ],
        dtype=float,
    )


def _configure_xaxis(ax, show_xaxis: bool):
    """Hide iteration-axis ticks/labels on upper rows to save vertical space."""
    if show_xaxis:
        return
    ax.set_xlabel("")
    ax.tick_params(axis="x", which="both", labelbottom=False, bottom=False, length=0)


STEP_TICK_ROUND = 5000


def _step_axis_range(steps, round_to=STEP_TICK_ROUND):
    """X-axis from 0 to ceil(max step / round_to) * round_to."""
    steps = np.asarray(steps, dtype=float)
    if steps.size == 0:
        return None
    lo = 0.0
    hi_raw = float(np.max(steps))
    hi = math.ceil(hi_raw / round_to) * round_to
    if hi <= lo:
        hi = float(round_to)
    return lo, hi


def _step_xticks(steps, n_ticks=5, round_to=STEP_TICK_ROUND):
    """Five ticks at 0, hi/4, hi/2, 3hi/4, hi with hi rounded up to a multiple of round_to."""
    steps = np.asarray(steps, dtype=float)
    if steps.size == 0:
        return [], None
    xlim = _step_axis_range(steps, round_to=round_to)
    if xlim is None:
        return [], None
    lo, hi = xlim
    ticks = [int(round(t)) for t in np.linspace(lo, hi, n_ticks)]
    ticks[0] = int(lo)
    ticks[-1] = int(hi)
    return ticks, (lo, hi)


def _sync_row_step_xaxis(axes, steps, show_xaxis: bool):
    """Same training-step x limits/ticks on every panel in a row (incl.\ empty)."""
    if not show_xaxis:
        return
    ticks, xlim = _step_xticks(steps)
    if xlim is None:
        return
    lo, hi = xlim
    for ax in axes:
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks)
        ax.tick_params(axis="x", which="both", labelbottom=True, bottom=True, length=3)


def _plot_series_on_axis(
    ax, steps, series, opts, empty=False, na_note=None, show_xaxis=True
):
    """Plot series; empty column keeps fixed width with no line."""
    if empty or series is None or not np.any(np.isfinite(series)):
        if opts.get("ylim"):
            ax.set_ylim(*opts["ylim"])
        _style_axis(ax)
        _configure_xaxis(ax, show_xaxis)
        if not show_xaxis:
            ax.set_xticks([])
        ax.set_yticks([])
        if na_note:
            ax.text(
                0.5,
                0.5,
                na_note,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="0.45",
            )
        return
    ax.plot(steps, series, color="C0")
    if opts.get("ylim"):
        ax.set_ylim(*opts["ylim"])
    _style_axis(ax)
    _configure_xaxis(ax, show_xaxis)


def _row_has_any_data(ts, row):
    gkey, lkey = row["gkey"], row["lkey"]
    yg = np.array(ts.get(gkey) or [], dtype=float)
    if np.any(np.isfinite(yg)):
        return True
    if not lkey:
        return False
    for li in range(row["n_data_layers"]):
        series = _layer_series(ts, lkey, li)
        if series is not None and np.any(np.isfinite(series)):
            return True
    return False


def _plot_intra_row(
    fig, gs, row_idx, n_data_cols, steps, ts, row, show_xlabel, row_letter=None
):
    gkey = row["gkey"]
    lkey = row["lkey"]
    is_pair = row["is_pair"]
    n_grid = row["n_grid_layers"]
    n_data = row["n_data_layers"]
    opts = _row_plot_opts(gkey)

    title_pad = 5 if show_xlabel else 3
    row_axes = []
    ax0 = fig.add_subplot(gs[row_idx, 0])
    row_axes.append(ax0)
    ax0.set_title("All", fontsize=9, pad=title_pad)
    yg = np.array(ts[gkey], dtype=float)
    _plot_series_on_axis(
        ax0, steps, yg, opts, empty=not np.any(np.isfinite(yg)), show_xaxis=show_xlabel
    )
    if _row_has_any_data(ts, row):
        ax0.set_ylabel(row["ylabel"])
    if show_xlabel:
        ax0.set_xlabel("Training step")

    if lkey and n_grid > 0:
        for li in range(n_grid):
            ax = fig.add_subplot(gs[row_idx, li + 1])
            row_axes.append(ax)
            na_note = _na_note_for_layer(gkey, li, n_grid)
            title = _layer_tick_label(li, is_pair, gkey=gkey, lkey=lkey)
            ax.set_title(title, fontsize=9, pad=title_pad)
            col_opts = _column_plot_opts(gkey, li, n_grid)
            if li >= n_data:
                _plot_series_on_axis(
                    ax,
                    steps,
                    None,
                    col_opts,
                    empty=True,
                    na_note=na_note,
                    show_xaxis=show_xlabel,
                )
            else:
                series = _layer_series(ts, lkey, li)
                has = series is not None and np.any(np.isfinite(series))
                if not has and na_note is None:
                    na_note = "no data"
                _plot_series_on_axis(
                    ax,
                    steps,
                    series,
                    col_opts,
                    empty=not has,
                    na_note=na_note,
                    show_xaxis=show_xlabel,
                )
            if show_xlabel:
                ax.set_xlabel("Training step")

    _sync_row_step_xaxis(row_axes, steps, show_xlabel)

    if row_letter is not None:
        ax_lbl = fig.add_subplot(gs[row_idx, n_data_cols])
        _plot_row_letter_axis(ax_lbl, row_letter)


def plot_stats_timeseries(
    ts,
    num_layers,
    out_path,
    epoch_metrics=None,
    dpi=150,
    label_rows=True,
    active_rows=None,
    all_active_rows=None,
    include_epoch=True,
    suptitle="SAL vs Backprop",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _apply_pub_style(dpi=dpi)

    steps = np.array(ts.get("step") or [], dtype=float)
    has_intra = len(steps) > 0
    all_active_rows = all_active_rows or (
        _active_intra_rows(ts, num_layers) if has_intra else []
    )
    if active_rows is None:
        active_rows = list(all_active_rows)
    n_intra = len(active_rows)
    has_epoch = (
        include_epoch and bool(epoch_metrics and epoch_metrics.get("epoch"))
    )

    if not has_epoch and n_intra == 0:
        raise ValueError("No epoch_metrics and no intra-epoch series to plot.")

    n_data_cols = 1 + num_layers
    n_cols = n_data_cols + (1 if label_rows else 0)
    width_ratios = [1.0] * n_data_cols
    if label_rows:
        width_ratios.append(0.055)

    # Uniform flat row aspect: wide figure, short row height, tight vertical gaps.
    row_h = 1.22
    epoch_h = 1.45 if has_epoch else 0.0
    intra_hspace = 0.22
    epoch_intra_hspace = EPOCH_INTRA_HSPACE
    fig_w = max(2.75 * n_data_cols, 10.0)

    intra_total_h = row_h * max(n_intra, 1)
    fig_h = (epoch_h if has_epoch else 0.0) + intra_total_h + 0.12
    fig = plt.figure(figsize=(fig_w, fig_h))

    letter_offset = 1 if (has_epoch and label_rows) else 0

    if has_epoch and n_intra == 0:
        fig.set_size_inches(max(fig_w, 9.0), max(epoch_h + 1.0, 4.0))
        gs_epoch = GridSpec(1, 1, figure=fig)
        _plot_epoch_row(
            fig,
            gs_epoch[0],
            n_data_cols,
            width_ratios,
            epoch_metrics,
            label_rows=label_rows,
            panel_wspace=0.36,
        )
        fig.subplots_adjust(left=0.12, top=0.92, bottom=0.14, right=0.98)
        _set_centered_suptitle(fig, suptitle, y=0.98)
        _savefig_pub(fig, out_path, dpi=dpi, pad_inches=0.04)
        plt.close(fig)
        return build_row_manifest(active_rows, letter_offset=letter_offset)

    if has_epoch:
        gs_outer = GridSpec(
            2,
            1,
            figure=fig,
            height_ratios=[epoch_h, intra_total_h],
            hspace=epoch_intra_hspace,
        )
        _plot_epoch_row(
            fig,
            gs_outer[0],
            n_data_cols,
            width_ratios,
            epoch_metrics,
            label_rows=label_rows,
        )
        gs_intra = gs_outer[1].subgridspec(
            n_intra,
            n_cols,
            height_ratios=[1.0] * n_intra,
            width_ratios=width_ratios,
            hspace=intra_hspace,
            wspace=0.30,
        )
    else:
        gs_intra = GridSpec(
            n_intra,
            n_cols,
            figure=fig,
            height_ratios=[1.0] * n_intra,
            width_ratios=width_ratios,
            hspace=intra_hspace,
            wspace=0.30,
        )

    for row_i, row in enumerate(active_rows):
        _plot_intra_row(
            fig,
            gs_intra,
            row_i,
            n_data_cols,
            steps,
            ts,
            row,
            show_xlabel=(row_i == n_intra - 1),
            row_letter=(
                _global_row_letter(row, active_rows, offset=letter_offset)
                if label_rows
                else None
            ),
        )
    fig.subplots_adjust(left=0.10, top=0.96, bottom=0.05, right=0.99)
    _set_centered_suptitle(fig, suptitle, y=0.995)
    _savefig_pub(fig, out_path, dpi=dpi, pad_inches=0.02)
    plt.close(fig)
    return build_row_manifest(active_rows, letter_offset=letter_offset)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot stat_timeseries.json from a SAL vs BP compare run."
    )
    p.add_argument("--exp_root", type=str, default="runs",
                   help="experiment root (same as training)")
    p.add_argument("--run_name", type=str, default=None,
                   help="run subdirectory under exp_root (optional if --input is set)")
    p.add_argument("--input", type=str, default=None,
                   help="path to stat_timeseries.json (default: run_dir/stat_timeseries.json)")
    p.add_argument("--output", type=str, default=None,
                   help="output figure path (default: run_dir/training_stats.png)")
    p.add_argument("--num_layers", type=int, default=None,
                   help="override layer count if missing from JSON")
    p.add_argument("--dpi", type=int, default=150, help="figure DPI")
    p.add_argument(
        "--no_row_labels",
        action="store_true",
        help="omit A, B, C, ... labels on the right of intra-epoch rows",
    )
    p.add_argument(
        "--row_manifest",
        type=str,
        default=None,
        help="write training_stats_rows.json (default: next to output figure)",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="write two figures: *_align.png (through cos V,V_bar) and *_updates.png",
    )
    p.add_argument(
        "--figure_part",
        type=str,
        default=None,
        choices=["align", "updates"],
        help="plot only one part (implies single output; use with --output)",
    )
    p.add_argument(
        "--core_only",
        action="store_true",
        help="only write the compact core alignment figure to --output (A–H)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.run_name is None and not args.input:
        raise SystemExit("Provide --run_name or --input")
    run_dir = (
        resolve_run_dir(args.exp_root, args.run_name)
        if args.run_name
        else os.path.dirname(os.path.abspath(args.input))
    )
    in_path = args.input or os.path.join(run_dir, "stat_timeseries.json")
    out_path = args.output or os.path.join(run_dir, "training_stats.png")

    ts, num_layers, _meta, epoch_metrics = load_timeseries(in_path, run_dir=run_dir)
    if args.num_layers is not None:
        num_layers = args.num_layers

    all_rows = _active_intra_rows(ts, num_layers) if ts.get("step") else []
    if args.core_only and args.figure_part is not None:
        raise SystemExit("Use only one of --core_only and --figure_part")
    if args.core_only:
        core_rows = _select_active_rows_by_gkeys(all_rows, _CORE_ROW_GKEYS)
        if not core_rows:
            raise SystemExit("No core alignment metrics found in this run.")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        manifest = plot_stats_timeseries(
            ts,
            num_layers,
            out_path,
            epoch_metrics=epoch_metrics,
            dpi=args.dpi,
            label_rows=not args.no_row_labels,
            active_rows=core_rows,
            all_active_rows=all_rows,
            include_epoch=True,
            suptitle="SAM Alignment Metrics",
        )
        mpath = args.row_manifest
        if mpath is None and manifest:
            pb, _ = os.path.splitext(out_path)
            mpath = f"{pb}_rows.json"
        if mpath and manifest:
            save_row_manifest(manifest, mpath)
            print(f"Wrote row manifest -> {mpath}")
        print(f"Wrote figure -> {out_path} ({len(core_rows)} core rows)")
        n_steps = len(ts.get("step") or [])
        n_ep = len((epoch_metrics or {}).get("epoch") or [])
        print(f"Read {n_steps} intra-epoch points, {n_ep} epochs from {in_path}")
        return

    do_split = args.split or args.figure_part is not None
    if do_split:
        parts = [args.figure_part] if args.figure_part else ["align", "updates"]
    else:
        parts = [None]

    base_out, ext = os.path.splitext(out_path)
    manifests = []

    def _emit_figure(
        subset,
        part_path,
        *,
        include_epoch,
        title,
        manifest_path=None,
    ):
        os.makedirs(os.path.dirname(os.path.abspath(part_path)) or ".", exist_ok=True)
        manifest = plot_stats_timeseries(
            ts,
            num_layers,
            part_path,
            epoch_metrics=epoch_metrics,
            dpi=args.dpi,
            label_rows=not args.no_row_labels,
            active_rows=subset,
            all_active_rows=all_rows,
            include_epoch=include_epoch,
            suptitle=title,
        )
        manifests.append((part_path, manifest))
        mpath = manifest_path
        if mpath is None and manifest:
            pb, _ = os.path.splitext(part_path)
            mpath = f"{pb}_rows.json"
        if mpath and manifest:
            save_row_manifest(manifest, mpath)
            print(f"Wrote row manifest -> {mpath}")
        print(f"Wrote figure -> {part_path} ({len(subset)} rows)")
        return manifest

    for part in parts:
        if part:
            subset = _split_active_rows(all_rows, part)
            if not subset:
                print(f"  skip empty part {part!r}")
                continue
            part_path = out_path if args.figure_part else f"{base_out}_{part}{ext}"
            include_epoch = part == "align"
            title = (
                "SAL vs Backprop (alignment)"
                if part == "align"
                else "SAL vs Backprop (updates)"
            )
        else:
            subset = all_rows
            part_path = out_path
            include_epoch = True
            title = "SAL vs Backprop"

        _emit_figure(
            subset,
            part_path,
            include_epoch=include_epoch,
            title=title,
            manifest_path=args.row_manifest,
        )

    # Compact core figure (always, unless --figure_part pins a single custom part).
    if args.figure_part is None:
        core_rows = _select_active_rows_by_gkeys(all_rows, _CORE_ROW_GKEYS)
        if core_rows:
            # Prefer vector PDF sibling when the main output is a raster image.
            core_ext = ext if ext.lower() in (".pdf", ".svg", ".eps") else ".pdf"
            core_path = f"{base_out}_core{core_ext}"
            _emit_figure(
                core_rows,
                core_path,
                include_epoch=True,
                title="SAM Alignment Metrics",
            )
            if ext.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                # Also keep a PNG preview next to the vector core PDF.
                _emit_figure(
                    core_rows,
                    f"{base_out}_core{ext}",
                    include_epoch=True,
                    title="SAM Alignment Metrics",
                )

    n_steps = len(ts.get("step") or [])
    n_ep = len((epoch_metrics or {}).get("epoch") or [])
    print(f"Read {n_steps} intra-epoch points, {n_ep} epochs from {in_path}")
    print(f"  total intra-epoch metric rows available: {len(all_rows)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
