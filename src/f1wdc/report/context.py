"""Assemble the template context for the HTML report."""

from __future__ import annotations

import pandas as pd

from f1wdc.config import SeasonConfig
from f1wdc.pipeline import Data, Hyper, Prediction
from f1wdc.report import html as H

KIND_LABEL = {
    "reg_year_2": ("second season of the 2026 technical rules", "rules year 2"),
    "reg_year_1": ("first season of a new technical rule set", "new rules"),
    "stable": ("a season with stable technical rules", "stable rules"),
}
SOURCE_NOTE = {"rookie": "rookie", "limited": "few starts", "returning": "returning"}


def driver_names(data: Data) -> dict[str, str]:
    d = data.drivers
    return {r.driverId: f"{r.givenName} {r.familyName}" for r in d.itertuples()}


def build_context(
    pred: Prediction,
    cfg: SeasonConfig,
    data: Data,
    hyper: Hyper,
    backtest_df: pd.DataFrame | None,
    backtest_drivers: pd.DataFrame | None,
    summary: pd.DataFrame | None,
    caveats: list[str],
    prev_cars: dict[str, float],
) -> dict:
    """``prev_cars``: end-of-previous-season car rating per lineage (centered on the grid)."""
    names = driver_names(data)
    team_names = dict(cfg.team_names)
    tab = pred.sim.driver_table()
    tab["name"] = [names.get(d, d) for d in tab["driverId"]]
    tab["team_name"] = [team_names.get(t, t) for t in tab["team"]]
    tab["note"] = [SOURCE_NOTE.get(pred.projection.source.get(d, ""), "") for d in tab["driverId"]]
    teams = pred.sim.team_table()
    teams["team_name"] = [team_names.get(t, t) for t in teams["team"]]

    prev_season = pred.target - 1
    params = pred.params
    pairs = params.car_pairs
    kind_label, kind_short = KIND_LABEL[pred.kind]
    as_of = (f"results through round {pred.as_of_round} of {pred.target}"
             if pred.as_of_round else f"all results through the {prev_season} season so far")
    n_rounds = max(e.round for e in pred.events) if pred.events else 0
    r2, r1, rs = params.rho_raw["reg_year_2"], params.rho_raw["reg_year_1"], params.rho_raw["stable"]
    if abs(r2 - rs) < 0.05:
        story = (f"Second-year seasons carry over about as much as stable ones ({r2:.2f} vs {rs:.2f} before "
                 f"shrinkage), clearly more than the first year of new rules ({r1:.2f}).")
    elif r2 > rs:
        story = (f"In the second year of a rule set the order holds up better than usual ({r2:.2f} vs {rs:.2f} "
                 f"in stable years, before shrinkage).")
    else:
        story = (f"Second-year seasons carry over a little less than stable ones ({r2:.2f} vs {rs:.2f} before "
                 f"shrinkage): teams that got the new rules wrong tend to catch up.")
    ctx = {
        "season": pred.target,
        "n_sims_text": f"{pred.sim.n_sims:,}",
        "as_of_text": as_of,
        "kind_label": kind_label,
        "kind_short": kind_short,
        "rho": pred.carryover,
        "rhos": params.rho,
        "n_pairs": params.n_pairs[pred.kind],
        "hero_name": tab.loc[0, "name"], "hero_team": tab.loc[0, "team_name"], "hero_p": tab.loc[0, "p_wdc"],
        "second_name": tab.loc[1, "name"], "second_p": tab.loc[1, "p_wdc"],
        "svg_wdc": H.wdc_bars(tab),
        "svg_heat": H.position_heatmap(tab, pred.sim.position_matrix(), pred.sim.drivers),
        "heat_legend": H.heat_legend(),
        "svg_car": H.car_pace(pred.projection.teams, team_names, pred.projection.car_mean(),
                              pred.projection.car_sd(), prev_cars),
        "svg_scatter": H.carry_scatter(pairs, params.rho),
        "carry_story": story,
        "drivers": tab.to_dict("records"),
        "teams": teams.to_dict("records"),
        "n_rounds": n_rounds,
        "n_sprints": sum(e.kind == "sprint" for e in pred.events),
        "caveats": caveats,
        "settings_text": (f"σ_driver {hyper.rating.sigma_driver}, σ_car {hyper.rating.sigma_car}, "
                          f"σ_rw {hyper.rating.sigma_rw}, outlier filter {hyper.rating.outlier_min:g} places, "
                          f"uncertainty ×{hyper.inflation}"),
    }
    if backtest_df is not None and summary is not None and backtest_drivers is not None:
        bt = backtest_df.copy()
        bt["champion_name"] = [names.get(d, d) for d in bt["champion"]]
        bt["top_pick_name"] = [names.get(d, d) for d in bt["top_pick"]]
        ctx.update(
            backtest=bt.to_dict("records"),
            summary=summary.to_dict("records"),
            svg_calib=H.calibration_chart(backtest_drivers),
            top_pick_rate=float((bt["rank_champion"] == 1).mean()),
            bt_seasons=len(bt),
        )
    else:
        ctx.update(backtest=[], summary=[], svg_calib="", top_pick_rate=float("nan"), bt_seasons=0)
    return ctx
