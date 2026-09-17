"""融合パラメータの掃引と leave-one-day-out。

GT を使ってパラメータを決めるので、そのままでは「調整に使った日でよく見える」
だけになりかねない。4日それぞれを順に保留し、残り3日で選んだ設定を保留日で
測り直すことで、過学習の幅を見る。

候補とスコアの読み込み、およびトラックの対応付け（`match_tracks`）は日付ごとに
**1回だけ**行って使い回す。掃引で変わるのは融合と割当だけで、そこは軽い。
対応付けはフレームごとの IoU を取るため最も重く、組み合わせの数だけ回すと
実行時間が桁で変わる。
"""

from __future__ import annotations

import collections
import itertools
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mct.appearance_sim import AppScore
from mct.assignment import assign_global_ids, solve
from mct.evaluation import evaluate
from mct.fusion import METHODS, app_value, fuse
from mct.predictions import Link
from mct.reid_pipeline import _load_scores
from mct.spatiotemporal import KIND_SELF_LOOP
from mct.st_pipeline import SCORES_FILENAME as ST_SCORES_FILENAME, load_candidates
from mct.appearance_sim import SCORES_FILENAME as APP_SCORES_FILENAME

__all__ = ["ALPHAS", "ALPHA_SELFS", "NO_MATCH_SCORES", "DayInputs", "run_sweep", "print_report"]

ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.7)
ALPHA_SELFS = (0.0, 0.1, 0.3, 0.5)
# 仕様の5水準に 0.50 / 0.60 / 0.70 を足す。誤ペアの融合スコアが中央 0.30・
# p90 0.44 なので、0.40 までだと最適値が範囲の端に出てしまう。
NO_MATCH_SCORES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)

METHOD = METHODS[6]


@dataclass
class DayInputs:
    """掃引のあいだ使い回す、1日分の読み込み済みデータ。"""

    date: str
    tracks: list[tuple[str, int]]
    candidates: list
    st: dict
    app: dict
    gt: Any
    matchings: dict


def _load_day(args, date: str) -> DayInputs:
    from mct.config import load_params
    from mct.cli import _load_gt, _matchings

    ev_params = load_params(args.params).get("evaluation") or {}
    summaries, candidates, _ = load_candidates(date, args.out)
    root = Path(args.out)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = _load_gt(args, date)

    return DayInputs(
        date=date,
        tracks=sorted(s.key for s in summaries),
        candidates=candidates,
        st=_load_scores(root / date / ST_SCORES_FILENAME, "likelihood"),
        app={
            k: AppScore.from_dict(v)
            for k, v in _load_scores(root / date / APP_SCORES_FILENAME, "").items()
        },
        gt=gt,
        matchings=_matchings(args, gt, date, ev_params),
    )


def _idf1(day: DayInputs, params: dict[str, Any]) -> float:
    """その設定での IDF1。融合と割当だけをやり直す。"""
    scored = {}
    for candidate in day.candidates:
        pair = (candidate.tgt, candidate.src)
        fused = fuse(
            day.st.get(pair), app_value(METHOD, day.app.get(pair)),
            is_self_loop=candidate.is_self_loop, method=METHOD, params=params,
        )
        if fused is not None:
            scored[pair] = fused[0]

    chosen = solve(day.tracks, scored, float(params["no_match_score"]))
    links = [Link(t[0], str(t[1]), s[0], str(s[1])) for t, s in chosen]
    assignments = assign_global_ids(links, day.tracks)
    return evaluate(assignments, day.gt, day.matchings).idf1


def _grid() -> list[dict[str, float]]:
    return [
        {"alpha": a, "alpha_self": s, "no_match_score": n}
        for a, s, n in itertools.product(ALPHAS, ALPHA_SELFS, NO_MATCH_SCORES)
    ]


def run_sweep(args, dates: list[str], report: Callable) -> None:
    print(f"{len(dates)} 日分を読み込み中...")
    days = {date: _load_day(args, date) for date in dates}

    grid = _grid()
    print(f"{len(grid)} 通り × {len(dates)} 日 を評価します\n")

    # 設定 -> 日付 -> IDF1
    table: dict[tuple[float, float, float], dict[str, float]] = {}
    for i, params in enumerate(grid, start=1):
        key = (params["alpha"], params["alpha_self"], params["no_match_score"])
        table[key] = {date: _idf1(days[date], params) for date in dates}
        if i % 40 == 0:
            print(f"  {i} / {len(grid)} 通り")

    report(table, dates, args)


def _best(table: dict, dates: list[str]) -> tuple[tuple[float, float, float], float]:
    scores = {k: statistics.mean(v[d] for d in dates) for k, v in table.items()}
    best = max(scores, key=scores.get)
    return best, scores[best]


def print_report(table: dict, dates: list[str], args) -> None:
    best, score = _best(table, dates)
    print("\n" + "=" * 78)
    print("融合パラメータの掃引（手法⑥）")
    print("=" * 78)
    print(f"  全{len(dates)}日で最良: α={best[0]} / α_self={best[1]}"
          f" / 未対応 {best[2]}  IDF1 {score:.4f}")

    # 各パラメータの周辺最大（他を最良値に固定して1つだけ振る）
    names = ("alpha", "alpha_self", "no_match_score")
    values = (ALPHAS, ALPHA_SELFS, NO_MATCH_SCORES)
    for axis, (name, candidates) in enumerate(zip(names, values)):
        print(f"\n  {name}（他は最良値に固定）")
        rows = []
        for v in candidates:
            key = tuple(v if i == axis else best[i] for i in range(3))
            if key in table:
                rows.append((v, statistics.mean(table[key][d] for d in dates)))
        top = max(rows, key=lambda r: r[1])[0] if rows else None
        for v, idf1 in rows:
            mark = "  ← 最大" if v == top else ""
            print(f"    {v:<8}{idf1:.4f}{mark}")
        if rows and top in (candidates[0], candidates[-1]):
            print(f"    ⚠ 最大値が掃引範囲の端（{top}）に出ています")

    _print_loo(table, dates)


def _print_loo(table: dict, dates: list[str]) -> None:
    """4日それぞれを保留した leave-one-day-out。"""
    print("\n" + "=" * 78)
    print("leave-one-day-out（3日で最適化 → 保留日で評価）")
    print("=" * 78)
    print(f"  {'保留日':<8}{'選ばれた設定':<28}{'学習時 IDF1':>12}{'保留日 IDF1':>13}{'差':>9}")
    print("  " + "-" * 72)

    train, held = [], []
    chosen: collections.Counter = collections.Counter()
    for holdout in dates:
        rest = [d for d in dates if d != holdout]
        if not rest:
            continue
        best, score = _best(table, rest)
        out = table[best][holdout]
        train.append(score)
        held.append(out)
        chosen[best] += 1
        label = f"α={best[0]} α_self={best[1]} 未対応={best[2]}"
        print(f"  {holdout:<8}{label:<28}{score:>12.4f}{out:>13.4f}{out - score:>+9.4f}")

    if train:
        print("  " + "-" * 72)
        print(f"  {'平均':<8}{'':<28}{statistics.mean(train):>12.4f}"
              f"{statistics.mean(held):>13.4f}{statistics.mean(held)-statistics.mean(train):>+9.4f}")
        if len(chosen) == 1:
            print(f"\n  4回とも同じ設定が選ばれました（{list(chosen)[0]}）。"
                  "パラメータは日付に依存していません。")
        else:
            print(f"\n  ⚠ 選ばれた設定が {len(chosen)} 通りに割れています: "
                  + ", ".join(f"{k}×{v}" for k, v in chosen.most_common()))
