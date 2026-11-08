"""Command-line entry point: ``f1wdc fetch | backtest | predict``."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from f1wdc.config import CONFIG_DIR

MODEL_CONFIG = CONFIG_DIR / "model.yaml"


def load_hyper(path: Path = MODEL_CONFIG):
    from f1wdc.model.ratings import RatingConfig
    from f1wdc.pipeline import Hyper

    if not path.exists():
        return Hyper()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rc = replace(RatingConfig(), **(data.get("rating") or {}))
    new_rules = data.get("inflation_new_rules")
    return Hyper(rc, float(data.get("inflation", 1.0)), float(data.get("dev_mult", 1.0)),
                 None if new_rules is None else float(new_rules))


def save_hyper(hyper, extra: dict, path: Path = MODEL_CONFIG) -> None:
    body = {"rating": asdict(hyper.rating), "inflation": hyper.inflation, "dev_mult": hyper.dev_mult,
            "inflation_new_rules": hyper.inflation_new_rules, **extra}
    header = "# Model settings chosen by `f1wdc backtest --tune` (grid search on 1997-2019 WDC log-loss).\n"
    path.write_text(header + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def cmd_fetch(args: argparse.Namespace) -> int:
    from f1wdc.data.build import build_all
    from f1wdc.data.jolpica import JolpicaClient, fetch_all

    client = JolpicaClient(cache_dir=Path(args.data_dir) / "raw")
    fetch_all(client, start=args.start, end=args.end, refresh=args.refresh,
              progress=(lambda msg: print(f"  fetching {msg}", flush=True)) if args.verbose else None)
    print(f"API requests made: {client.requests_made}")
    print(build_all(Path(args.data_dir)))
    return 0


def _backtest(data, hyper, last_year: int, n_sims: int):
    from f1wdc.backtest import TRAIN_YEARS, add_baselines, run_backtest, summarize, tune_share_temperature

    years = [y for y in range(1997, last_year + 1) if data.calendar(y)]
    df, drivers = run_backtest(data, hyper, years, n_sims=n_sims)
    df = add_baselines(data, df, tune_share_temperature(data, [y for y in TRAIN_YEARS if y in years]))
    return df, drivers, summarize(df)


def cmd_backtest(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    from f1wdc.backtest import tune, tune_inseason
    from f1wdc.pipeline import Data

    console = Console()
    data = Data.load(args.data_dir)
    hyper = load_hyper()
    if args.tune:
        console.print("Tuning on 1997-2019 (holdout 2020+ untouched)...")
        hyper, res = tune(data, n_sims=args.tune_sims, progress=lambda m: console.print(f"  {m}"))
        Path(args.out).mkdir(parents=True, exist_ok=True)
        res.to_csv(Path(args.out) / "tuning_results.csv", index=False)
        console.print("Tuning in-season development drift (predictions from 1/3, 1/2, 3/4 of each season)...")
        mult, res_in = tune_inseason(data, hyper, n_sims=args.tune_sims, progress=lambda m: console.print(f"  {m}"))
        hyper = replace(hyper, dev_mult=mult)
        save_hyper(hyper, {"tuning_logloss_1997_2019": round(float(res.iloc[0]["logloss"]), 4),
                           "inseason_logloss_1997_2019": round(float(res_in.iloc[0]["logloss"]), 4)})
        console.print(f"Saved best settings to {MODEL_CONFIG}")
    df, drivers, summary = _backtest(data, hyper, args.to, args.sims)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / "backtest.csv", index=False)
    drivers.to_csv(Path(args.out) / "backtest_drivers.csv", index=False)

    t = Table(title="Backtest by season")
    for c in ("season", "type", "champion", "P(champion)", "rank", "favourite", "P(fav)"):
        t.add_column(c, justify="right" if c.startswith("P") or c == "rank" else "left")
    for r in df.itertuples():
        t.add_row(str(r.season), r.type, r.champion, f"{r.p_champion:.1%}", str(r.rank_champion), r.top_pick, f"{r.p_top_pick:.1%}")
    console.print(t)
    s = Table(title="Mean WDC log-loss (lower is better)")
    for c in ("group", "seasons", "model", "points share", "top-2 teams", "champion top pick", "spearman"):
        s.add_column(c)
    for r in summary.itertuples():
        s.add_row(r.group, str(r.seasons), f"{r.logloss_model:.3f}", f"{r.logloss_points_share:.3f}",
                  f"{r.logloss_top2_teams:.3f}", f"{r.champion_top_pick:.0%}", f"{r.spearman:.2f}")
    console.print(s)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    from f1wdc.config import load_season_config
    from f1wdc.pipeline import Data, compute_ratings, predict
    from f1wdc.report.context import build_context, driver_names
    from f1wdc.report.html import render_report

    console = Console()
    if args.update:
        cmd_fetch(argparse.Namespace(data_dir=args.data_dir, start=1994, end=None, refresh=False, verbose=False))
    data = Data.load(args.data_dir)
    hyper = load_hyper()
    cfg = load_season_config(args.grid)
    season = cfg.season
    api_cal = data.calendar(season)
    n_rounds, sprints = api_cal if api_cal else (cfg.n_rounds, cfg.sprint_rounds)
    ratings = compute_ratings(data, hyper)
    pred = predict(data, ratings, season, cfg.grid, n_rounds, sprints, cfg.race_points, cfg.sprint_points,
                   hyper, n_sims=args.sims, seed=args.seed, carryover=args.carryover,
                   dob_overrides=cfg.dob_overrides)

    names = driver_names(data)
    tab = pred.sim.driver_table()
    t = Table(title=f"{season} World Drivers' Championship - {pred.sim.n_sims:,} simulated seasons "
                    f"(car carry-over {pred.carryover:.2f}, {pred.kind})")
    for c in ("#", "driver", "team", "P(title)", "P(top 3)", "exp. pts", "80% range"):
        t.add_column(c, justify="left" if c in ("driver", "team") else "right")
    for i, r in enumerate(tab.itertuples(), 1):
        t.add_row(str(i), names.get(r.driverId, r.driverId), cfg.team_names.get(r.team, r.team), f"{r.p_wdc:.1%}",
                  f"{r.p_top3:.1%}", f"{r.exp_points:.0f}", f"{r.p10:.0f}–{r.p90:.0f}")
    console.print(t)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tab.assign(name=[names.get(d, d) for d in tab["driverId"]]).to_csv(out / f"predictions_{season}.csv", index=False)
    pred.sim.team_table().to_csv(out / f"constructors_{season}.csv", index=False)

    if args.no_report:
        return 0
    bt = (None, None, None)
    if not args.no_backtest:
        console.print("Running backtest for the report (1997 onward)...")
        bt = _backtest(data, hyper, season - 1, args.backtest_sims)
    _, cars = ratings
    prev = cars[(cars["season"] == season - 1) & cars["lineage"].isin(cfg.grid)].set_index("lineage")["c_end"]
    prev = (prev - prev.mean()).to_dict()
    last_round = data.completed_rounds(season - 1)
    total_prev = data.calendar(season - 1)[0] if data.calendar(season - 1) else last_round
    caveats = [
        f"Built on {season - 1} results through round {last_round} of {total_prev}. Re-run "
        f"'f1wdc predict --update' after later races; during {season} it conditions on races already run.",
        "Only about five seasons in history were the second year of a new rule set, so the carry-over rate is uncertain. "
        "Try '--carryover 0.6' or '--carryover 1.0' to see how much the answer moves.",
        "Rule details are not modelled directly: power-unit catch-up allowances for manufacturers that are behind (ADUO), "
        "budget cap and wind-tunnel time only enter through the historical carry-over rate.",
        "The model rates pace from finishing order. It does not reward converting good positions into wins, and "
        "2026 retirement data does not separate mechanical failures from crashes.",
        "Driver line-ups come from config/season_2027.yaml (Haas: Câmara replaces Ocon; Bearman's option not yet "
        "officially announced). Sprint round numbers are approximate until the API publishes the calendar.",
    ]
    ctx = build_context(pred, cfg, data, hyper, *bt, caveats=caveats, prev_cars=prev)
    path = render_report(ctx, out / f"report_{season}.html")
    console.print(f"Report: {path.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="f1wdc", description="F1 World Drivers' Championship predictor")
    p.add_argument("--data-dir", default="data")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download/refresh Jolpica data and rebuild tables")
    f.add_argument("--start", type=int, default=1994)
    f.add_argument("--end", type=int, default=None)
    f.add_argument("--refresh", action="store_true", help="force refetch of the current season")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="predict past seasons from prior data and score them")
    b.add_argument("--to", type=int, default=2026, help="last season to backtest")
    b.add_argument("--sims", type=int, default=5000)
    b.add_argument("--tune", action="store_true", help="grid-search model settings on 1997-2019 first")
    b.add_argument("--tune-sims", type=int, default=2000)
    b.add_argument("--out", default="out")
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("predict", help="simulate a season and write the report")
    r.add_argument("--grid", default=str(CONFIG_DIR / "season_2027.yaml"))
    r.add_argument("--sims", type=int, default=20_000)
    r.add_argument("--seed", type=int, default=2027)
    r.add_argument("--carryover", type=float, default=None, help="override the learned car carry-over rate")
    r.add_argument("--update", action="store_true", help="refresh data from the API first")
    r.add_argument("--no-backtest", action="store_true", help="skip the backtest section of the report")
    r.add_argument("--backtest-sims", type=int, default=5000)
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--out", default="out")
    r.set_defaults(func=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252 (no ρ, é, ü)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
