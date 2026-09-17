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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mct.appearance_sim import AppScore
from mct.assignment import assign_global_ids, solve
from mct.evaluation import evaluate
from mct.fusion import METHODS, app_value, fuse
from mct.predictions import Link
from mct.reid_pipeline import _load_scores
from mct.config import load_cameras
from mct.spatiotemporal import (
    KIND_IN_RANGE, KIND_OUT_OF_RANGE, KIND_SELF_LOOP, dwell_constant, room_zones,
)
from mct.st_pipeline import SCORES_FILENAME as ST_SCORES_FILENAME, load_candidates
from mct.appearance_sim import SCORES_FILENAME as APP_SCORES_FILENAME

__all__ = ["ALPHAS", "ALPHA_SELFS", "NO_MATCH_SCORES", "DayInputs", "run_sweep", "print_report"]

# 仕様の5水準に 0.0 / 0.05 を足す。最初の掃引で下端の 0.1 が最大になり、
# IDF1 が α に対して単調減少していたため、外側まで見ないと最大値を言えない。
# α=0 は「時空間を使わない」= 手法②と同じ設定にあたる。
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
ALPHA_SELFS = (0.0, 0.1, 0.3, 0.5, 0.7)
# 仕様の5水準に 0.50 / 0.60 / 0.70 を足す。誤ペアの融合スコアが中央 0.30・
# p90 0.44 なので、0.40 までだと最適値が範囲の端に出てしまう。
NO_MATCH_SCORES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)

# c_dwell = exp(-k^2/2) の k。α=0 では c_dwell が結果に効かないので、
# k の掃引では α の下限を 0 ではなく「時空間を少しだけ混ぜる」側に置く。
DWELL_KS = (1.5, 2.0, 2.5, 3.0, 3.5)
DWELL_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.5)

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
    correct: set = field(default_factory=set)
    kinds: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    """候補ごとの誤差フレーム。None なら到着時刻を予測できず c_dwell に落ちた。"""
    room: dict = field(default_factory=dict)
    """候補の退出ゾーンが居室につながっているか。"""


def _load_day(args, date: str) -> DayInputs:
    from mct.config import load_params
    from mct.cli import _correct_pairs_and_summaries, _load_gt, _matchings

    ev_params = load_params(args.params).get("evaluation") or {}
    summaries, candidates, _ = load_candidates(date, args.out)
    root = Path(args.out)
    rooms = room_zones(load_cameras(args.config))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = _load_gt(args, date)

    return DayInputs(
        correct=_correct_pairs_and_summaries(args, date, summaries)[0],
        kinds=_load_scores(root / date / ST_SCORES_FILENAME, "kind"),
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
        errors=_load_scores(root / date / ST_SCORES_FILENAME, "error_frames"),
        room={
            (c.tgt, c.src): (
                c.route is not None
                and (c.route.from_cam, c.route.from_zone) in rooms
            )
            for c in candidates
        },
    )


def _st_value(day: DayInputs, pair, params: dict[str, Any]) -> float | None:
    """その設定での時空間尤度。

    k を振ると `c_dwell` が変わるので、`st_scores.json` の値をそのまま
    使えるのは「範囲内で到着時刻を予測できた候補」だけになる。残りは
    k から導出し直す。誤差が入っている候補は k に依存しない。
    """
    kind = day.kinds.get(pair)
    if kind is None:
        return None
    if "dwell_k_self" not in params:
        return day.st.get(pair)          # k を振らない掃引ではそのまま

    if kind == KIND_SELF_LOOP:
        return dwell_constant(float(params["dwell_k_self"]))
    if kind == KIND_IN_RANGE and day.errors.get(pair) is not None:
        return day.st.get(pair)          # 誤差から出した値は k に依らない
    k = params["dwell_k_room"] if day.room.get(pair) else params["dwell_k_no_room"]
    return dwell_constant(float(k))


def _run(day: DayInputs, params: dict[str, Any]) -> tuple[float, dict[str, tuple[int, int]]]:
    """(IDF1, 種類ごとの (統合できた数, 正解数))。融合と割当だけをやり直す。"""
    scored = {}
    for candidate in day.candidates:
        pair = (candidate.tgt, candidate.src)
        fused = fuse(
            _st_value(day, pair, params), app_value(METHOD, day.app.get(pair)),
            is_self_loop=candidate.is_self_loop, method=METHOD, params=params,
        )
        if fused is not None:
            scored[pair] = fused[0]

    chosen = solve(day.tracks, scored, float(params["no_match_score"]))
    links = [Link(t[0], str(t[1]), s[0], str(s[1])) for t, s in chosen]
    assignments = assign_global_ids(links, day.tracks)

    taken = set(chosen)
    counts: dict[str, tuple[int, int]] = {}
    for pair in day.correct:
        kind = day.kinds.get(pair)
        if kind is None:
            continue
        ok, total = counts.get(kind, (0, 0))
        counts[kind] = (ok + int(pair in taken), total + 1)
    return evaluate(assignments, day.gt, day.matchings).idf1, counts


def _idf1(day: DayInputs, params: dict[str, Any]) -> float:
    return _run(day, params)[0]


def _grid() -> list[dict[str, float]]:
    return [
        {"alpha": a, "alpha_self": s, "no_match_score": n}
        for a, s, n in itertools.product(ALPHAS, ALPHA_SELFS, NO_MATCH_SCORES)
    ]


def _dwell_grid(no_match_score: float) -> list[dict[str, float]]:
    """k と α の格子。

    `alpha_self` は α と同じにする。自己ループの重みを別に持たせると
    `dwell_k_self` と同じところを二重に動かすことになり、どちらが効いたのか
    読めなくなる。
    """
    return [
        {
            "alpha": a, "alpha_self": a, "no_match_score": no_match_score,
            "dwell_k_room": kr, "dwell_k_no_room": kn, "dwell_k_self": ks,
        }
        for a, kr, kn, ks in itertools.product(
            DWELL_ALPHAS, DWELL_KS, DWELL_KS, DWELL_KS
        )
    ]


def run_sweep(args, dates: list[str], report: Callable) -> None:
    print(f"{len(dates)} 日分を読み込み中...")
    days = {date: _load_day(args, date) for date in dates}

    dwell_mode = getattr(args, "dwell", False)
    grid = _dwell_grid(float(getattr(args, "no_match_score", None) or 0.10)) \
        if dwell_mode else _grid()
    print(f"{len(grid)} 通り × {len(dates)} 日 を評価します\n")

    # 設定 -> 日付 -> IDF1
    table: dict[tuple, dict[str, float]] = {}
    kinds: dict[tuple, dict[str, tuple[int, int]]] = {}
    for i, params in enumerate(grid, start=1):
        key = (
            (params["alpha"], params["dwell_k_room"], params["dwell_k_no_room"],
             params["dwell_k_self"])
            if dwell_mode
            else (params["alpha"], params["alpha_self"], params["no_match_score"])
        )
        per_date, totals = {}, collections.defaultdict(lambda: [0, 0])
        for date in dates:
            idf1, counts = _run(days[date], params)
            per_date[date] = idf1
            for kind, (ok, total) in counts.items():
                totals[kind][0] += ok
                totals[kind][1] += total
        table[key] = per_date
        kinds[key] = {k: tuple(v) for k, v in totals.items()}
        if i % 40 == 0:
            print(f"  {i} / {len(grid)} 通り")

    if dwell_mode:
        print_dwell_report(table, dates, kinds)
    else:
        report(table, dates, args, kinds)


def _best(table: dict, dates: list[str]) -> tuple[tuple[float, float, float], float]:
    scores = {k: statistics.mean(v[d] for d in dates) for k, v in table.items()}
    best = max(scores, key=scores.get)
    return best, scores[best]


def print_report(table: dict, dates: list[str], args, kinds: dict | None = None) -> None:
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
        # 0 は α の自然な下限なので、端に出ても「外側を見るべき」ではない
        at_edge = rows and top in (candidates[0], candidates[-1]) and top != 0.0
        if at_edge:
            print(f"    ⚠ 最大値が掃引範囲の端（{top}）に出ています")

    if kinds:
        _print_top(table, dates, kinds)
    _print_loo(table, dates)


def _print_top(table: dict, dates: list[str], kinds: dict, n: int = 8) -> None:
    """IDF1 の高い設定を、種類ごとの正解率と並べて出す。

    IDF1 だけを見ると、件数の多い in_range を優先して自己ループを丸ごと
    捨てる設定が「最良」に見えることがある。少数派を切り捨てていないかを
    同じ表で確かめられるようにする。
    """
    scores = {k: statistics.mean(v[d] for d in dates) for k, v in table.items()}
    order = sorted(scores, key=scores.get, reverse=True)[:n]

    print("\n" + "=" * 78)
    print(f"IDF1 の高い設定 上位{n}件と、種類ごとの正解率")
    print("=" * 78)
    print(f"  {'α':>5}{'α_self':>8}{'未対応':>8}{'IDF1':>9}"
          f"{'in_range':>12}{'out_of_range':>14}{'self_loop':>12}")
    print("  " + "-" * 68)
    for key in order:
        cells = ""
        for kind in (KIND_IN_RANGE, KIND_OUT_OF_RANGE, KIND_SELF_LOOP):
            ok, total = kinds[key].get(kind, (0, 0))
            width = {KIND_IN_RANGE: 12, KIND_OUT_OF_RANGE: 14, KIND_SELF_LOOP: 12}[kind]
            cells += f"{f'{ok}/{total} ({100*ok/max(1,total):.0f}%)':>{width}}"
        print(f"  {key[0]:>5}{key[1]:>8}{key[2]:>8}{scores[key]:>9.4f}{cells}")

    dropped = [k for k in order if kinds[k].get(KIND_SELF_LOOP, (0, 0))[0] == 0]
    if dropped:
        print(f"\n  ⚠ 上位のうち {len(dropped)} 件は自己ループを1件も拾えていません。"
              "\n    IDF1 はわずかに上がりますが、自己ループは本研究の対象なので"
              "\n    この設定を採るなら意図的な取捨であることを明示する必要があります。")


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


# --------------------------------------------------------------------------
# k の掃引の報告
# --------------------------------------------------------------------------

DWELL_AXES = ("α", "k_room", "k_no_room", "k_self")


def print_dwell_report(table: dict, dates: list[str], kinds: dict, n: int = 12) -> None:
    """c_dwell の k と α を振った結果。

    IDF1 だけでなく種類ごとの正解率を並べる。狙いは out_of_range の正解を
    拾えるようにすることなので、IDF1 が同じでも out_of_range が上がる組が
    あればそちらを採りたい。
    """
    scores = {k: statistics.mean(v[d] for d in dates) for k, v in table.items()}
    best = max(scores, key=scores.get)

    print("\n" + "=" * 86)
    print("c_dwell の k と α の掃引（手法⑥）")
    print("=" * 86)
    print(f"  最良: α={best[0]} / k_room={best[1]} / k_no_room={best[2]}"
          f" / k_self={best[3]}  IDF1 {scores[best]:.4f}")

    _print_axes(table, dates, scores, best)

    # α=0 では時空間の項が消えるので k は結果に一切効かない。k の効き方を
    # 見るには α>0 に絞る必要がある。
    positive = {k: v for k, v in scores.items() if k[0] > 0}
    if positive:
        best_positive = max(positive, key=positive.get)  # noqa: F841
        print(f"\n  --- α>0 に絞った場合（α=0 では k が結果に効かないため）---")
        print(f"  最良: α={best_positive[0]} / k_room={best_positive[1]}"
              f" / k_no_room={best_positive[2]} / k_self={best_positive[3]}"
              f"  IDF1 {positive[best_positive]:.4f}")
        _print_axes(table, dates, scores, best_positive, skip_alpha=True)

    _print_top_table("IDF1 の高い設定 上位", scores, kinds, n)
    if positive:
        # α=0 の行が上位を埋めるので、時空間を混ぜた側も別に並べる
        _print_top_table("α>0 に絞った上位", positive, kinds, n)

    _best_for_out_of_range(scores, kinds)
    if positive:
        _best_for_out_of_range(positive, kinds, label="（α>0 に絞った場合）")


def _print_top_table(title: str, scores: dict, kinds: dict, n: int) -> None:
    print(f"\n  {title}{n}件と種類ごとの正解率")
    print(f"  {'α':>5}{'k_room':>8}{'k_noroom':>10}{'k_self':>8}{'IDF1':>9}"
          f"{'in_range':>12}{'out_of_range':>14}{'self_loop':>12}")
    print("  " + "-" * 78)
    for key in sorted(scores, key=scores.get, reverse=True)[:n]:
        cells = ""
        for kind, width in (
            (KIND_IN_RANGE, 12), (KIND_OUT_OF_RANGE, 14), (KIND_SELF_LOOP, 12)
        ):
            ok, total = kinds[key].get(kind, (0, 0))
            cells += f"{f'{ok}/{total} ({100*ok/max(1,total):.0f}%)':>{width}}"
        print(f"  {key[0]:>5}{key[1]:>8}{key[2]:>10}{key[3]:>8}"
              f"{scores[key]:>9.4f}{cells}")


def _print_axes(table: dict, dates: list[str], scores: dict, base: tuple,
                skip_alpha: bool = False) -> None:
    """各軸を1つだけ振ったときの IDF1。他は `base` に固定する。"""
    values = (DWELL_ALPHAS, DWELL_KS, DWELL_KS, DWELL_KS)
    for axis, (name, candidates) in enumerate(zip(DWELL_AXES, values)):
        if skip_alpha and axis == 0:
            continue
        print(f"\n  {name}（他は固定）")
        rows = []
        for v in candidates:
            key = tuple(v if i == axis else base[i] for i in range(4))
            if key in table:
                rows.append((v, statistics.mean(table[key][d] for d in dates)))
        top = max(rows, key=lambda r: r[1])[0] if rows else None
        flat = rows and max(r[1] for r in rows) - min(r[1] for r in rows) < 1e-9
        for v, idf1 in rows:
            mark = "" if flat else ("  ← 最大" if v == top else "")
            print(f"    {v:<8}{idf1:.4f}{mark}")
        if flat:
            print("    （この軸は結果に影響していません）")


def _best_for_out_of_range(scores: dict, kinds: dict, label: str = "") -> None:
    """out_of_range の正解率が最も高い設定と、そのときの IDF1。"""
    def rate(key, kind):
        ok, total = kinds[key].get(kind, (0, 0))
        return ok / total if total else 0.0

    best_out = max(scores, key=lambda k: (rate(k, KIND_OUT_OF_RANGE), scores[k]))
    best_idf1 = max(scores, key=scores.get)
    print(f"\n  out_of_range の正解率が最大{label}: α={best_out[0]} / k_room={best_out[1]}"
          f" / k_no_room={best_out[2]} / k_self={best_out[3]}")
    print(f"    out_of_range {100*rate(best_out, KIND_OUT_OF_RANGE):.0f}%"
          f" / self_loop {100*rate(best_out, KIND_SELF_LOOP):.0f}%"
          f" / IDF1 {scores[best_out]:.4f}"
          f"（IDF1 最良との差 {scores[best_out]-scores[best_idf1]:+.4f}）")
