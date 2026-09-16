"""時空間尤度がどれだけ正解と誤ペアを分けられているかの診断。

本フェーズの成否はここで測る。AUC が 0.5 に近ければ、時空間尤度は
機能していない。外観を足したときの改善幅を測る基準線にもなる。
"""

from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mct.spatiotemporal import KIND_IN_RANGE, KIND_OUT_OF_RANGE, KIND_SELF_LOOP, STScore

__all__ = ["Labelled", "auc", "print_separation", "print_kind_breakdown",
           "print_error_by_zone", "print_ranks", "plot_separation"]

KINDS = (KIND_IN_RANGE, KIND_OUT_OF_RANGE, KIND_SELF_LOOP)
PNG_NAME = "st_likelihood_separation.png"


@dataclass
class Labelled:
    """正解かどうかの印を付けたスコア。"""

    score: STScore
    correct: bool
    date: str = ""
    zone_pair: tuple[str, str, str, str] | None = None


def auc(positive: list[float], negative: list[float]) -> float:
    """正解の尤度が誤ペアより高い確率。同値は0.5として数える。

    総当たりは O(n·m) になるので、順位和（Mann-Whitney U）から求める。
    """
    if not positive or not negative:
        return float("nan")

    values = np.concatenate([np.asarray(positive), np.asarray(negative)])
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)

    # 同値には平均順位を割り当てる（同値を0.5として数えるのと等価）
    sorted_values = values[order]
    start = 0
    for i in range(1, len(sorted_values) + 1):
        if i == len(sorted_values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    n_pos, n_neg = len(positive), len(negative)
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _quantiles(values: list[float]) -> str:
    if not values:
        return "-"
    a = np.asarray(values)
    return (
        f"{np.percentile(a,10):.3f} / {np.percentile(a,25):.3f} / {np.median(a):.3f} / "
        f"{np.percentile(a,75):.3f} / {np.percentile(a,90):.3f}"
    )


def print_separation(labelled: list[Labelled]) -> float:
    """正解と誤ペアの尤度分布、および AUC。"""
    pos = [x.score.likelihood for x in labelled if x.correct]
    neg = [x.score.likelihood for x in labelled if not x.correct]
    overall = auc(pos, neg)

    print("\n" + "=" * 72)
    print("尤度の分離度（本フェーズの成否指標）")
    print("=" * 72)
    print(f"  候補 {len(labelled)} 件（正解 {len(pos)} / 誤 {len(neg)}）")
    print(f"  正解ペア  p10/p25/中央/p75/p90: {_quantiles(pos)}")
    print(f"  誤ペア    p10/p25/中央/p75/p90: {_quantiles(neg)}")
    print(f"\n  AUC（全候補）: {overall:.3f}")

    # 全体の AUC には「探索時間範囲に入るか否か」の効果が混ざるので、
    # 種類の中だけで見た値も出す。誤差そのものの効き具合が分かる。
    for kind in KINDS:
        p = [x.score.likelihood for x in labelled if x.correct and x.score.kind == kind]
        n = [x.score.likelihood for x in labelled if not x.correct and x.score.kind == kind]
        value = auc(p, n)
        note = ""
        if not np.isnan(value):
            if value < 0.5:
                note = "  ← 偶然以下"
            elif value < 0.6:
                note = "  ← ほぼ偶然"
        print(f"  AUC（{kind:<13}）: {value:.3f}  (正解 {len(p)} / 誤 {len(n)}){note}")
    return overall


def print_kind_breakdown(labelled: list[Labelled]) -> None:
    print("\n種類ごとの内訳")
    print(f"  {'種類':<14}{'候補数':>8}{'正解':>7}{'正解率':>8}   尤度 p10/p25/中央/p75/p90")
    print("  " + "-" * 78)
    for kind in KINDS:
        items = [x for x in labelled if x.score.kind == kind]
        correct = [x for x in items if x.correct]
        values = [x.score.likelihood for x in items]
        print(
            f"  {kind:<14}{len(items):>8}{len(correct):>7}"
            f"{100*len(correct)/max(1,len(items)):>7.1f}%   {_quantiles(values)}"
        )

    missed = [x for x in labelled if x.correct and x.score.kind == KIND_OUT_OF_RANGE]
    print(f"\n  探索時間範囲を超えた候補に含まれる正解: {len(missed)} 件")
    if missed:
        print("    （固定値 c_dwell が当たるため、誤差の情報は使われない）")
        for x in missed[:6]:
            s = x.score
            print(f"      {x.date} {s.tgt[0]}#{s.tgt[1]} -> {s.src[0]}#{s.src[1]}"
                  f"  尤度 {s.likelihood:.2f}")
        if len(missed) > 6:
            print(f"      ... 他 {len(missed) - 6} 件")


def print_error_by_zone(labelled: list[Labelled], fps: float) -> None:
    """ゾーン対ごとの誤差の中央値。距離校正済みなので0付近のはず。"""
    by_zone: dict[tuple[str, str, str, str], list[float]] = collections.defaultdict(list)
    for x in labelled:
        if x.correct and x.score.error_frames is not None and x.zone_pair:
            by_zone[x.zone_pair].append(x.score.error_frames)

    print("\n正解ペアの誤差（ゾーン対ごと・件数の多い順）")
    if not by_zone:
        print("  対象がありません")
        return

    warned = []
    for key, values in sorted(by_zone.items(), key=lambda kv: -len(kv[1]))[:12]:
        median = statistics.median(values)
        mark = ""
        if abs(median) > fps:                 # 1秒以上ずれていれば目立たせる
            mark = "  ← 1秒以上のずれ"
            warned.append((key, median))
        print(
            f"  {key[0]}/{key[1]} -> {key[2]}/{key[3]:<12} {len(values):>3} 件  "
            f"中央 {median:+6.0f} フレーム ({median/fps:+.2f} 秒){mark}"
        )
    if warned:
        print(f"\n  ⚠ 中央値が1秒以上ずれているゾーン対が {len(warned)} 対あります"
              "（距離か速度にまだバイアスがある可能性）")


def print_ranks(labelled: list[Labelled]) -> None:
    """src トラックごとに尤度で並べ、正解が何位に来るか。

    ID 統合の前段階として、時空間尤度だけでどこまで絞れるかを示す。
    """
    by_src: dict[tuple[str, tuple[str, int]], list[Labelled]] = collections.defaultdict(list)
    for x in labelled:
        by_src[(x.date, x.score.src)].append(x)

    ranks: list[tuple[int, int, Labelled]] = []      # (順位, 候補数, 正解)
    for items in by_src.values():
        correct = [x for x in items if x.correct]
        if not correct:
            continue
        ordered = sorted(items, key=lambda x: -x.score.likelihood)
        for target in correct:
            rank = next(i for i, x in enumerate(ordered, 1) if x is target)
            ranks.append((rank, len(ordered), target))

    print("\n正解ペアの順位（src トラックごとに尤度の降順）")
    if not ranks:
        print("  対象がありません")
        return

    total = len(ranks)
    for n in (1, 3, 5):
        hit = sum(1 for r, _, _ in ranks if r <= n)
        print(f"  {n}位以内: {hit:>3} / {total} = {100*hit/total:.1f}%")
    positions = [r for r, _, _ in ranks]
    print(f"  順位の中央値 {int(statistics.median(positions))} / 最悪 {max(positions)}")
    print(f"  候補数の中央値 {int(statistics.median([n for _, n, _ in ranks]))} 件/src")

    worst = sorted(ranks, key=lambda t: -(t[0] / max(1, t[1])))[:6]
    print("\n  正解が下位に沈んだ例:")
    for rank, n, x in worst:
        s = x.score
        print(
            f"    {x.date} {s.tgt[0]}#{s.tgt[1]} -> {s.src[0]}#{s.src[1]}"
            f"  {rank}/{n} 位  尤度 {s.likelihood:.3f}  種類 {s.kind}"
        )


def plot_separation(labelled: list[Labelled], dest: str | Path) -> Path:
    """正解と誤ペアの尤度を重ねたヒストグラム。"""
    import matplotlib
    matplotlib.use("Agg")                      # 画面が無い環境で描く
    import matplotlib.pyplot as plt

    pos = [x.score.likelihood for x in labelled if x.correct]
    neg = [x.score.likelihood for x in labelled if not x.correct]

    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_all, ax_in) = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = np.linspace(0, 1, 41)
    for ax, subset, title in (
        (ax_all, labelled, "all candidates"),
        (ax_in, [x for x in labelled if x.score.kind == KIND_IN_RANGE], "in_range only"),
    ):
        p = [x.score.likelihood for x in subset if x.correct]
        n = [x.score.likelihood for x in subset if not x.correct]
        # 件数が2桁違うので割合で重ねる
        ax.hist(n, bins=bins, density=True, alpha=0.55, label=f"wrong (n={len(n)})",
                color="#888888")
        ax.hist(p, bins=bins, density=True, alpha=0.65, label=f"correct (n={len(p)})",
                color="#d1495b")
        ax.set_title(f"{title}   AUC={auc(p, n):.3f}")
        ax.set_xlabel("spatiotemporal likelihood")
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Spatiotemporal likelihood separation  "
        f"(correct {len(pos)} / wrong {len(neg)})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
