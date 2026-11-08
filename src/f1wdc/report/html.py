"""Standalone HTML report with inline SVG charts and hover tooltips (light + dark)."""

from __future__ import annotations

import html
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES = Path(__file__).parent / "templates"
LABEL_W = 190          # left label column in row charts (viewBox units)
CHART_W = 760


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pct(p: float) -> str:
    if p <= 0:
        return "0%"
    if p < 0.005:
        return "<1%"
    return f"{p * 100:.0f}%" if p >= 0.095 else f"{p * 100:.1f}%"


def _bar_path(x0: float, x1: float, y: float, h: float, r: float = 4) -> str:
    if x1 - x0 < r:
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 0):.1f}" height="{h:.1f}"/>'
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z"/>')


def _row_label(y: float, name: str, sub: str) -> str:
    return (f'<text class="lbl" x="0" y="{y:.1f}">{esc(name)}'
            f'<tspan class="sub" dx="6">{esc(sub)}</tspan></text>')


def wdc_bars(tab: pd.DataFrame) -> str:
    row_h, bar_h = 26, 14
    x0, x1 = LABEL_W, CHART_W - 56
    top = max(float(tab["p_wdc"].max()), 0.01)
    step = next(s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25) if top / s <= 6)
    vmax = float(np.ceil(top / step - 1e-9) * step)
    h = row_h * len(tab) + 24
    out = [f'<svg viewBox="0 0 {CHART_W} {h}" role="img" aria-label="Championship win probability by driver">']
    for v in np.arange(0, vmax + 1e-9, step):
        x = x0 + v / vmax * (x1 - x0)
        out.append(f'<line class="grid{" base" if v == 0 else ""}" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{h - 22}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{h - 6}" text-anchor="middle">{v * 100:.0f}%</text>')
    for i, r in enumerate(tab.itertuples()):
        y = i * row_h
        xe = x0 + (x1 - x0) * r.p_wdc / vmax
        tip = (f"{r.name} ({r.team_name})\nWins title: {pct(r.p_wdc)}\nTop 3: {pct(r.p_top3)}\n"
               f"Expected points: {r.exp_points:.0f}\n80% range: {r.p10:.0f}–{r.p90:.0f}")
        out.append(f'<g class="row" data-tip="{esc(tip)}" tabindex="0">'
                   f'<rect class="hit" x="0" y="{y}" width="{CHART_W}" height="{row_h}"/>'
                   + _row_label(y + row_h / 2 + 4, r.name, r.team_name)
                   + f'<g class="s1">{_bar_path(x0, xe, y + (row_h - bar_h) / 2, bar_h)}</g>'
                   f'<text class="val" x="{xe + 6:.1f}" y="{y + row_h / 2 + 4:.1f}">{pct(r.p_wdc)}</text></g>')
    out.append("</svg>")
    return "".join(out)


HEAT_EDGES = [0.005, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]


def position_heatmap(tab: pd.DataFrame, pos_matrix: np.ndarray, drivers: list[str]) -> str:
    n = pos_matrix.shape[1]
    row_h, gap = 20, 2
    x0 = LABEL_W
    cw = (CHART_W - x0) / n
    h = row_h * len(tab) + 22
    di = {d: i for i, d in enumerate(drivers)}
    out = [f'<svg viewBox="0 0 {CHART_W} {h}" role="img" aria-label="Probability of each final championship position">']
    for j in range(n):
        if j == 0 or (j + 1) % 5 == 0:
            out.append(f'<text class="tick" x="{x0 + j * cw + cw / 2:.1f}" y="{h - 6}" text-anchor="middle">P{j + 1}</text>')
    for i, r in enumerate(tab.itertuples()):
        y = i * row_h
        out.append(_row_label(y + row_h / 2 + 4, r.name, ""))
        probs = pos_matrix[di[r.driverId]]
        for j, p in enumerate(probs):
            bucket = int(np.searchsorted(HEAT_EDGES, p, side="right"))
            if bucket == 0:
                continue
            tip = f"{r.name}\nFinishes P{j + 1}: {pct(p)}"
            out.append(f'<rect class="h{bucket}" x="{x0 + j * cw + gap / 2:.1f}" y="{y + gap / 2}" width="{cw - gap:.1f}" '
                       f'height="{row_h - gap}" rx="2" data-tip="{esc(tip)}"/>')
    out.append("</svg>")
    return "".join(out)


def heat_legend() -> str:
    labels = ["<2%", "2–5%", "5–10%", "10–20%", "20–35%", "35–50%", "50%+"]
    return "".join(f'<span class="key"><svg width="14" height="14"><rect class="h{i + 1}" width="14" height="14" rx="2"/></svg>{esc(l)}</span>'
                   for i, l in enumerate(labels))


def car_pace(teams: list[str], names: dict[str, str], mean: np.ndarray, sd: np.ndarray, prev: dict[str, float]) -> str:
    order = np.argsort(-mean)
    row_h = 30
    z = 1.2816
    lo = min(float((mean - z * sd).min()), min(prev.values(), default=0.0)) - 0.2
    hi = max(float((mean + z * sd).max()), max(prev.values(), default=0.0)) + 0.2
    x0, x1 = LABEL_W, CHART_W - 20
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    h = row_h * len(teams) + 24
    out = [f'<svg viewBox="0 0 {CHART_W} {h}" role="img" aria-label="Projected 2027 car pace by team">']
    for t in np.arange(np.ceil(lo * 2) / 2, hi + 1e-9, 0.5):
        out.append(f'<line class="grid{" base" if abs(t) < 1e-9 else ""}" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="0" y2="{h - 22}"/>')
        out.append(f'<text class="tick" x="{sx(t):.1f}" y="{h - 6}" text-anchor="middle">{t:+.1f}</text>')
    for i, k in enumerate(order):
        t = teams[k]
        y = i * row_h + row_h / 2
        m, s = mean[k], sd[k]
        p26 = prev.get(t)
        tip = (f"{names.get(t, t)}\n2027 projection: {m:+.2f} (80%: {m - z * s:+.2f} to {m + z * s:+.2f})"
               + (f"\nEnd of 2026: {p26:+.2f}" if p26 is not None else "\nNew team"))
        out.append(f'<g class="row" data-tip="{esc(tip)}" tabindex="0"><rect class="hit" x="0" y="{y - row_h / 2}" width="{CHART_W}" height="{row_h}"/>'
                   + _row_label(y + 4, names.get(t, t), "")
                   + f'<line class="s1-line" x1="{sx(m - z * s):.1f}" x2="{sx(m + z * s):.1f}" y1="{y}" y2="{y}"/>')
        if p26 is not None:
            out.append(f'<circle class="s2 ring" cx="{sx(p26):.1f}" cy="{y}" r="4.5"/>')
        out.append(f'<circle class="s1 ring" cx="{sx(m):.1f}" cy="{y}" r="5.5"/></g>')
    out.append("</svg>")
    return "".join(out)


def carry_scatter(pairs: pd.DataFrame, rho: dict[str, float]) -> str:
    W, H = CHART_W, 420
    pad_l, pad_b, pad_t, pad_r = 48, 34, 10, 16
    lim = float(np.ceil(max(pairs["c_end"].abs().max(), pairs["c_next"].abs().max()) * 2) / 2)
    sx = lambda v: pad_l + (v + lim) / (2 * lim) * (W - pad_l - pad_r)
    sy = lambda v: pad_t + (lim - v) / (2 * lim) * (H - pad_t - pad_b)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Car rating carry-over from one season to the next">']
    for t in np.arange(-lim, lim + 1e-9, 0.5 if lim <= 2 else 1.0):
        cls = "grid base" if abs(t) < 1e-9 else "grid"
        out.append(f'<line class="{cls}" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{pad_t}" y2="{H - pad_b}"/>')
        out.append(f'<line class="{cls}" x1="{pad_l}" x2="{W - pad_r}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        out.append(f'<text class="tick" x="{sx(t):.1f}" y="{H - pad_b + 16}" text-anchor="middle">{t:+.1f}</text>')
        out.append(f'<text class="tick" x="{pad_l - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:+.1f}</text>')
    out.append(f'<text class="axis-title" x="{W - pad_r}" y="{H - 4}" text-anchor="end">car rating at end of season N →</text>')
    out.append(f'<text class="axis-title" x="{pad_l + 4}" y="{pad_t + 12}">↑ car rating in season N+1</text>')
    style = {"stable": "s0", "reg_year_1": "s2", "reg_year_2": "s1"}
    label = {"stable": "Stable rules", "reg_year_1": "New rules (year 1)", "reg_year_2": "Second year of rules"}
    for kind in ("stable", "reg_year_1", "reg_year_2"):
        sub = pairs[pairs["type"] == kind]
        for r in sub.itertuples():
            tip = f"{r.lineage}: {r.season} → {r.season + 1}\n{label[kind]}\nEnd {r.season}: {r.c_end:+.2f}\n{r.season + 1}: {r.c_next:+.2f}"
            out.append(f'<circle class="{style[kind]} ring" cx="{sx(r.c_end):.1f}" cy="{sy(r.c_next):.1f}" r="4" data-tip="{esc(tip)}"/>')
    for kind in ("stable", "reg_year_1", "reg_year_2"):
        k = rho[kind]
        a, b = -lim, lim
        ya, yb = np.clip(k * a, -lim, lim), np.clip(k * b, -lim, lim)
        out.append(f'<line class="{style[kind]}-line" x1="{sx(ya / k if k else a):.1f}" y1="{sy(ya):.1f}" '
                   f'x2="{sx(yb / k if k else b):.1f}" y2="{sy(yb):.1f}"/>')
    out.append("</svg>")
    return "".join(out)


def calibration_chart(drv: pd.DataFrame) -> str:
    edges = [0, 0.05, 0.15, 0.3, 0.5, 0.7, 1.0001]
    drv = drv.assign(bin=pd.cut(drv["p_top3"], edges, right=False))
    g = drv.groupby("bin", observed=True).agg(pred=("p_top3", "mean"), obs=("actual_top3", "mean"), n=("p_top3", "size"))
    W, H, pad = 360, 300, 40
    s = lambda v: pad + v * (W - 2 * pad)
    sy = lambda v: H - pad - v * (H - 2 * pad)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Calibration of predicted top-3 probability">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line class="grid" x1="{s(t):.1f}" x2="{s(t):.1f}" y1="{sy(0):.1f}" y2="{sy(1):.1f}"/>')
        out.append(f'<line class="grid" x1="{s(0):.1f}" x2="{s(1):.1f}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        out.append(f'<text class="tick" x="{s(t):.1f}" y="{H - pad + 16}" text-anchor="middle">{t:.0%}</text>')
        out.append(f'<text class="tick" x="{pad - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:.0%}</text>')
    out.append(f'<line class="ref" x1="{s(0)}" y1="{sy(0)}" x2="{s(1)}" y2="{sy(1)}"/>')
    pts = " ".join(f"{s(r.pred):.1f},{sy(r.obs):.1f}" for r in g.itertuples())
    out.append(f'<polyline class="s1-line" points="{pts}" fill="none"/>')
    for r in g.itertuples():
        tip = f"Predicted top-3 chance: {r.pred:.0%}\nActually finished top 3: {r.obs:.0%}\nDriver-seasons: {r.n}"
        out.append(f'<circle class="s1 ring" cx="{s(r.pred):.1f}" cy="{sy(r.obs):.1f}" r="5" data-tip="{esc(tip)}"/>')
    out.append(f'<text class="axis-title" x="{W - pad}" y="{H - 6}" text-anchor="end">predicted →</text>')
    out.append(f'<text class="axis-title" x="{pad + 4}" y="{pad - 8}">↑ observed</text>')
    out.append("</svg>")
    return "".join(out)


def render_report(ctx: dict, path: Path) -> Path:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    env.filters["pct"] = pct
    page = env.get_template("report.html.j2").render(generated=date.today().isoformat(), **ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
