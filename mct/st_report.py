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

from mct.metrics import auc, format_quantiles
from mct.spatiotemporal import KIND_IN_RANGE, KIND_OUT_OF_RANGE, KIND_SELF_LOOP, STScore

__all__ = [
    "Labelled", "auc", "print_separation", "print_kind_breakdown",
    "print_error_by_zone", "print_competitors", "print_ranks",
    "print_self_loop_ranks", "plot_separation",
]

KINDS = (KIND_IN_RANGE, KIND_OUT_OF_RANGE, KIND_SELF_LOOP)
PNG_NAME = "st_likelihood_separation.png"


@dataclass
class Labelled:
    """正解かどうかの印を付けたスコア。"""

    score: STScore
    correct: bool
    date: str = ""
    zone_pair: tuple[str, str, str, str] | None = None


def print_separation(labelled: list[Labelled]) -> float:
    """正解と誤ペアの尤度分布、および AUC。"""
    pos = [x.score.likelihood for x in labelled if x.correct]
    neg = [x.score.likelihood for x in labelled if not x.correct]
    overall = auc(pos, neg)

    print("\n" + "=" * 72)
    print("尤度の分離度（本フェーズの成否指標）")
    print("=" * 72)
    print(f"  候補 {len(labelled)} 件（正解 {len(pos)} / 誤 {len(neg)}）")
    print(f"  正解ペア  p10/p25/中央/p75/p90: {format_quantiles(pos)}")
    print(f"  誤ペア    p10/p25/中央/p75/p90: {format_quantiles(neg)}")
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
            f"{100*len(correct)/max(1,len(items)):>7.1f}%   {format_quantiles(values)}"
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


def _group_by_src(
    labelled: list[Labelled],
) -> dict[tuple[str, tuple[str, int]], list[Labelled]]:
    grouped: dict[tuple[str, tuple[str, int]], list[Labelled]] = collections.defaultdict(list)
    for x in labelled:
        grouped[(x.date, x.score.src)].append(x)
    return grouped


def _ranks(items: list[Labelled], target: Labelled) -> tuple[float, float]:
    """(平均順位, 同値を最下位とした順位)。

    自己ループは尤度が `c_dwell` 固定で同値だらけになるので、同値の扱いで
    見え方が変わる。平均順位は無作為に順序を付けたときの期待値、最下位扱いは
    最悪の場合にあたる。両方を出して読み手が判断できるようにする。
    """
    higher = sum(1 for x in items if x.score.likelihood > target.score.likelihood)
    tied = sum(1 for x in items if x.score.likelihood == target.score.likelihood)
    return higher + (tied + 1) / 2.0, float(higher + tied)


def _histogram(counts: list[int], label: str) -> None:
    buckets = collections.Counter(min(c, 5) for c in counts)
    total = max(1, len(counts))
    cells = " ".join(
        f"{('5+' if k == 5 else str(k))}:{buckets.get(k, 0):>4}" for k in range(6)
    )
    print(f"  {label:<16}{cells}   （中央値 {int(statistics.median(counts))}）"
          if counts else f"  {label:<16}（対象なし）")


def print_competitors(labelled: list[Labelled]) -> None:
    """src トラックごとの競合数の分布。

    順位は分母が何件かで意味が変わる。候補が1件しかなければ、正解が1位に
    なるのは自明で、尤度の寄与ではない。
    """
    by_src = _group_by_src(labelled)
    totals, in_range, out_range, self_loop = [], [], [], []
    for items in by_src.values():
        totals.append(len(items))
        in_range.append(sum(1 for x in items if x.score.kind == KIND_IN_RANGE))
        out_range.append(sum(1 for x in items if x.score.kind == KIND_OUT_OF_RANGE))
        self_loop.append(sum(1 for x in items if x.score.kind == KIND_SELF_LOOP))

    print(f"\nsrc トラックごとの候補数の分布（{len(by_src)} トラック / 件数ごとの度数）")
    print(f"  {'':16}{'0件':>6} {'1件':>6} {'2件':>6} {'3件':>6} {'4件':>6} {'5件以上':>6}")
    _histogram(totals, "候補総数")
    _histogram(in_range, "in_range")
    _histogram(out_range, "out_of_range")
    _histogram(self_loop, "self_loop")

    only_one = sum(1 for n in in_range if n == 1)
    none = sum(1 for n in in_range if n == 0)
    two_plus = sum(1 for n in in_range if n >= 2)
    print(
        f"\n  in_range が 0件: {none} / 1件のみ: {only_one} / 2件以上: {two_plus}"
    )
    print(
        f"  ※ in_range が1件だけの {only_one} トラックでは、正解が1位になるのは自明で"
        "、尤度の寄与ではない"
    )


def _print_rank_block(
    title: str, ranks: list[tuple[float, float, int, Labelled]]
) -> None:
    if not ranks:
        print(f"\n  {title}: 対象がありません")
        return

    total = len(ranks)
    print(f"\n  {title}（{total} 件）")
    for n in (1, 3, 5):
        hit = sum(1 for r, _, _, _ in ranks if r <= n)
        print(f"    {n}位以内: {hit:>3} / {total} = {100*hit/total:.1f}%")
    positions = [r for r, _, _, _ in ranks]
    print(
        f"    順位の中央値 {statistics.median(positions):.1f} / 最悪 {max(positions):.1f}"
        f" / 分母の中央値 {int(statistics.median([n for _, _, n, _ in ranks]))} 件"
    )


def print_ranks(labelled: list[Labelled]) -> None:
    """正解ペアの順位を3通りの分母で出す。

    全候補で数えると、大半を占める `out_of_range`（尤度は c_dwell 固定で
    範囲内の正解より確実に低い）が分母に入り、順位が尤度の性能ではなく
    トポロジーと時間窓の絞り込みを映してしまう。範囲内だけ、さらに競合が
    生じた場合だけ、と絞って見る。
    """
    by_src = _group_by_src(labelled)

    all_ranks: list[tuple[float, float, int, Labelled]] = []
    in_ranks: list[tuple[float, float, int, Labelled]] = []
    contested: list[tuple[float, float, int, Labelled]] = []
    by_competitors: dict[int, list[bool]] = collections.defaultdict(list)
    excluded = collections.Counter()

    for items in by_src.values():
        in_items = [x for x in items if x.score.kind == KIND_IN_RANGE]
        for target in (x for x in items if x.correct):
            avg, worst = _ranks(items, target)
            all_ranks.append((avg, worst, len(items), target))

            if target.score.kind != KIND_IN_RANGE:
                excluded[target.score.kind] += 1      # 範囲内での順位は定義できない
                continue

            avg_in, worst_in = _ranks(in_items, target)
            in_ranks.append((avg_in, worst_in, len(in_items), target))
            by_competitors[min(len(in_items), 4)].append(avg_in <= 1.0)
            if len(in_items) >= 2:
                contested.append((avg_in, worst_in, len(in_items), target))

    print("\n正解ペアの順位（src トラックごとに尤度の降順・同値は平均順位）")
    _print_rank_block("① 全候補を分母とする（参考値）", all_ranks)
    _print_rank_block("② in_range のみを分母とする", in_ranks)
    _print_rank_block("③ in_range が2件以上ある src のみ（本質的な指標）", contested)

    if excluded:
        detail = " / ".join(f"{k} {n} 件" for k, n in sorted(excluded.items()))
        print(
            f"\n  ②③ から除外した正解: {sum(excluded.values())} 件（{detail}）"
            "\n    正解が in_range に無いので、範囲内での順位を定義できないため"
        )

    print("\n  in_range の競合数ごとの1位率")
    print(f"    {'候補数':<8}{'src数':>7}{'正解が1位':>10}{'1位率':>9}")
    for key in sorted(by_competitors):
        values = by_competitors[key]
        hits = sum(values)
        label = "4以上" if key == 4 else str(key)
        print(f"    {label:<8}{len(values):>7}{hits:>10}{100*hits/len(values):>8.1f}%")

    worst_cases = sorted(contested, key=lambda t: -(t[0] / max(1, t[2])))[:6]
    if worst_cases:
        print("\n  競合があって正解が下位に沈んだ例:")
        for avg, _, n, x in worst_cases:
            s = x.score
            print(
                f"    {x.date} {s.tgt[0]}#{s.tgt[1]} -> {s.src[0]}#{s.src[1]}"
                f"  {avg:.1f}/{n} 位  尤度 {s.likelihood:.3f}"
            )


def print_self_loop_ranks(labelled: list[Labelled]) -> None:
    """自己ループだけの順位。

    尤度が `c_dwell` 固定なので、同じゾーン対の自己ループ同士は完全に同値で
    順位が付かない。平均順位と最下位扱いの両方を出して、その影響を示す。
    """
    by_src = _group_by_src(labelled)
    ranks: list[tuple[float, float, int, Labelled]] = []
    n_tied_at_top = 0

    for items in by_src.values():
        loops = [x for x in items if x.score.kind == KIND_SELF_LOOP]
        for target in (x for x in loops if x.correct):
            avg, worst = _ranks(loops, target)
            ranks.append((avg, worst, len(loops), target))
            best = max(x.score.likelihood for x in loops)
            if target.score.likelihood == best and sum(
                1 for x in loops if x.score.likelihood == best
            ) > 1:
                n_tied_at_top += 1

    print("\n自己ループの順位（自己ループ候補のみを分母とする）")
    if not ranks:
        print("  対象がありません")
        return

    counts = [n for _, _, n, _ in ranks]
    print(f"  正解 {len(ranks)} 件 / 分母の中央値 {int(statistics.median(counts))} 件"
          f" / 最大 {max(counts)} 件")
    print(f"  自己ループ候補が1件だけ: {sum(1 for n in counts if n == 1)} 件"
          f" / 2件以上: {sum(1 for n in counts if n >= 2)} 件")

    for label, index in (("平均順位", 0), ("同値を最下位とした順位", 1)):
        hits = sum(1 for r in ranks if r[index] <= 1.0)
        median = statistics.median([r[index] for r in ranks])
        print(f"  {label:<22} 1位 {hits:>3} / {len(ranks)} = "
              f"{100*hits/len(ranks):.1f}%  中央値 {median:.1f}")

    print(f"  最高値で同値だった正解: {n_tied_at_top} 件"
          "（順位が尤度では決まらず、どれを選ぶかは後段しだい）")


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
