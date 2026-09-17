"""再同定の結果の診断。

IDF1 だけでは「どこを直せば上がるのか」が分からない。採用された対応と
取りこぼした正解をそれぞれ理由で分類するのが本モジュールの主目的で、
候補列挙の限界なのか、順位付けの誤りなのか、対応すべきでない対を
取ってしまったのかを切り分ける。

`in_range` / `out_of_range` / `self_loop` を分けて見るのは、後ろ2つが
本研究の対象だからである。滞留や入室をまたぐ再同定と、同じ部屋へ戻る
自己ループは、探索時間の範囲内で素直につながる対応とは難しさが違う。
"""

from __future__ import annotations

import collections
import statistics
from typing import Any

from mct.evaluation import EvalResult
from mct.gt import GTTransition
from mct.predictions import Link
from mct.reid_pipeline import DayResult
from mct.spatiotemporal import KIND_IN_RANGE, KIND_OUT_OF_RANGE, KIND_SELF_LOOP

__all__ = [
    "KINDS", "LinkErrors", "classify_links", "print_results", "print_error_breakdown",
    "print_kind_accuracy", "print_component_sizes", "print_cost", "GT_COMPONENT_SIZES",
]

KINDS = (KIND_IN_RANGE, KIND_OUT_OF_RANGE, KIND_SELF_LOOP)

# GT の Global ID あたりのトラック数（4日合計・実測）
GT_COMPONENT_SIZES = {1: 5, 2: 86, 3: 65, 4: 10, 5: 2}

TrackKey = tuple[str, int]

# 採用された対応の分類
LINK_CORRECT = "正解"
LINK_RANK = "順位の誤り"
LINK_ENUM = "候補列挙の限界"
LINK_SPURIOUS = "未対応にすべきだった"

# 取りこぼした正解の分類
MISS_ENUM = "正解が候補に無かった"
MISS_RANK = "候補にあったが選ばれなかった"
MISS_NONE = "未対応と判定された"


class LinkErrors:
    """リンク単位の誤りの集計。

    トラック単位の IDFP/IDFN とは粒度が違う。1本のリンクの誤りが2トラックの
    ID を狂わせることもあれば、連鎖して波及することもあるので、件数は
    一致しない。改善点を指すのはこちらで、性能を測るのは IDF1 の側になる。
    """

    def __init__(self) -> None:
        self.links: collections.Counter = collections.Counter()
        self.misses: collections.Counter = collections.Counter()
        self.examples: dict[str, list[str]] = collections.defaultdict(list)

    def add_link(self, label: str, detail: str = "") -> None:
        self.links[label] += 1
        if detail and len(self.examples[label]) < 5:
            self.examples[label].append(detail)

    def add_miss(self, label: str, detail: str = "") -> None:
        self.misses[label] += 1
        if detail and len(self.examples[label]) < 5:
            self.examples[label].append(detail)


def classify_links(
    result: DayResult,
    correct: set[tuple[TrackKey, TrackKey]],
    errors: LinkErrors | None = None,
) -> LinkErrors:
    """採用された対応と、取りこぼした正解を理由で分類する。"""
    errors = errors or LinkErrors()

    # 候補に入っていて採点もできた対（= 割当が選べた対）
    scorable = set(result.scored)
    chosen = {(link.from_key, link.to_key) for link in result.predictions.links}

    # その src（後に出現する側）について、正解がどれだったか
    correct_by_src = {src: tgt for tgt, src in correct}

    for link in result.predictions.links:
        pair = (link.from_key, link.to_key)
        src = link.to_key
        detail = (
            f"{result.date} {link.from_cam}#{link.from_track}"
            f" -> {link.to_cam}#{link.to_track} S={link.score:.3f}"
        )
        if pair in correct:
            errors.add_link(LINK_CORRECT)
        elif src not in correct_by_src:
            errors.add_link(LINK_SPURIOUS, detail)
        elif (correct_by_src[src], src) in scorable:
            errors.add_link(LINK_RANK, detail)
        else:
            errors.add_link(LINK_ENUM, detail)

    for tgt, src in sorted(correct - chosen):
        detail = f"{result.date} {tgt[0]}#{tgt[1]} -> {src[0]}#{src[1]}"
        if (tgt, src) not in scorable:
            errors.add_miss(MISS_ENUM, detail)
        elif any(s == src for _, s in chosen):
            errors.add_miss(MISS_RANK, detail)
        else:
            errors.add_miss(MISS_NONE, detail)
    return errors


def print_results(rows: list[tuple[int, str, EvalResult]]) -> None:
    """7a。手法ごとに全日付を合算した ID 指標。"""
    from mct.fusion import METHODS

    by_method: dict[int, list[EvalResult]] = collections.defaultdict(list)
    for method, _, result in rows:
        by_method[method].append(result)

    print("\n" + "=" * 96)
    print("ID 指標（トラック単位・全日付の合算）")
    print("=" * 96)
    header = (
        f"  {'手法':<4}{'構成':<26}{'IDTP':>6}{'IDFP':>6}{'IDFN':>6}"
        f"{'IDP':>8}{'IDR':>8}{'IDF1':>8}{'GID 正解率':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) + 12))

    for method in sorted(by_method):
        results = by_method[method]
        total = EvalResult(date="all")
        total.idtp = sum(r.idtp for r in results)
        total.idfp = sum(r.idfp for r in results)
        total.idfn = sum(r.idfn for r in results)
        n_gid = sum(r.n_gt_global_ids for r in results)
        n_ok = sum(r.n_gt_global_ids_correct for r in results)
        print(
            f"  {method:<4}{METHODS[method].label:<26}{total.idtp:>6}{total.idfp:>6}"
            f"{total.idfn:>6}{total.idp:>8.3f}{total.idr:>8.3f}{total.idf1:>8.3f}"
            f"{f'{n_ok}/{n_gid} ({100*n_ok/max(1,n_gid):.1f}%)':>16}"
        )


def print_error_breakdown(
    errors: LinkErrors, n_links: int, n_correct: int, result: EvalResult | None = None
) -> None:
    """7c。リンク単位の誤りの分類。"""
    print("\n" + "=" * 78)
    print("誤りの分類（リンク単位）")
    print("=" * 78)
    if result is not None:
        print(f"  ※ トラック単位の IDFP {result.idfp} / IDFN {result.idfn} とは粒度が違う。"
              "\n     1本のリンクの誤りが複数トラックの ID を狂わせるため件数は一致しない。")

    print(f"\n  採用した対応 {n_links} 件")
    for label in (LINK_CORRECT, LINK_RANK, LINK_ENUM, LINK_SPURIOUS):
        n = errors.links.get(label, 0)
        print(f"    {label:<24}{n:>5} 件 ({100*n/max(1,n_links):.1f}%)")
        for example in errors.examples.get(label, [])[:3]:
            print(f"      例: {example}")

    n_missed = sum(errors.misses.values())
    print(f"\n  取りこぼした正解 {n_missed} / {n_correct} 件")
    for label in (MISS_ENUM, MISS_RANK, MISS_NONE):
        n = errors.misses.get(label, 0)
        print(f"    {label:<24}{n:>5} 件 ({100*n/max(1,n_correct):.1f}%)")
        for example in errors.examples.get(label, [])[:3]:
            print(f"      例: {example}")


def print_kind_accuracy(counts: dict[str, tuple[int, int]]) -> None:
    """7d。種類ごとの正解数と、そのうち正しく統合できた数。"""
    print("\n" + "=" * 78)
    print("種類ごとの正解率")
    print("=" * 78)
    print(f"  {'種類':<16}{'正解':>7}{'統合できた':>12}{'正解率':>10}")
    print("  " + "-" * 50)
    for kind in KINDS:
        ok, total = counts.get(kind, (0, 0))
        print(f"  {kind:<16}{total:>7}{ok:>12}{100*ok/max(1,total):>9.1f}%")
    ok = sum(v[0] for v in counts.values())
    total = sum(v[1] for v in counts.values())
    print("  " + "-" * 50)
    print(f"  {'合計':<16}{total:>7}{ok:>12}{100*ok/max(1,total):>9.1f}%")
    print("\n  ※ out_of_range（滞留・入室をまたぐ遷移）と self_loop（同じ部屋へ戻る）"
          "\n     が本研究の対象。in_range は探索時間の範囲内で素直につながる対応。")


def print_component_sizes(assignments: list[dict[str, dict[str, str]]]) -> None:
    """7e。予測された Global ID が持つトラック数の分布。"""
    sizes: collections.Counter = collections.Counter()
    for day in assignments:
        per_gid: dict[str, int] = collections.Counter()
        for tracks in day.values():
            for global_id in tracks.values():
                per_gid[global_id] += 1
        for n in per_gid.values():
            sizes[n] += 1

    print("\n" + "=" * 78)
    print("連結成分のサイズ分布（Global ID あたりのトラック数）")
    print("=" * 78)
    keys = sorted(set(sizes) | set(GT_COMPONENT_SIZES))
    print(f"  {'トラック数':<12}" + "".join(f"{k:>7}" for k in keys) + f"{'合計':>8}")
    print("  " + "-" * (12 + 7 * len(keys) + 8))
    for label, table in (("GT", GT_COMPONENT_SIZES), ("予測", sizes)):
        row = "".join(f"{table.get(k, 0):>7}" for k in keys)
        print(f"  {label:<12}{row}{sum(table.values()):>8}")

    oversized = {k: v for k, v in sizes.items() if k > max(GT_COMPONENT_SIZES)}
    if oversized:
        print(f"\n  ⚠ GT の最大（{max(GT_COMPONENT_SIZES)}台）を超える連結成分: "
              + ", ".join(f"{k}台 {v}件" for k, v in sorted(oversized.items()))
              + "\n    併合の誤りが連鎖すると、多数のトラックが1つの ID にまとまる。")


def print_cost(results: list[DayResult]) -> None:
    """7f。候補数と計算時間。"""
    by_method: dict[int, list[DayResult]] = collections.defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    print("\n" + "=" * 86)
    print("候補数と計算時間（全日付の合計 / 1日あたりの中央値）")
    print("=" * 86)
    print(f"  {'手法':<5}{'トラック':>9}{'候補':>9}{'採点':>9}{'採用':>7}"
          f"{'候補列挙':>10}{'類似度':>9}{'求解':>9}")
    print("  " + "-" * 74)
    for method in sorted(by_method):
        days = by_method[method]
        print(
            f"  {method:<5}{sum(d.n_tracks for d in days):>9}"
            f"{sum(d.n_candidates for d in days):>9}"
            f"{sum(len(d.scored) for d in days):>9}"
            f"{sum(d.n_links for d in days):>7}"
            f"{sum(d.sec_candidates for d in days):>9.2f}秒"
            f"{sum(d.sec_similarity for d in days):>8.2f}秒"
            f"{sum(d.sec_solve for d in days):>8.2f}秒"
        )
    solves = [d.sec_solve for d in results]
    if solves:
        print(f"\n  割当の求解: 1日あたり中央 {statistics.median(solves)*1000:.1f} ミリ秒"
              f" / 最大 {max(solves)*1000:.1f} ミリ秒")


def gt_kind_counts(
    result: DayResult,
    correct: set[tuple[TrackKey, TrackKey]],
    counts: dict[str, tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    """種類ごとの (統合できた数, 正解数) を足し込む。

    種類は候補の側に付くので、候補に無い正解は数えられない。分母を
    「候補に入った正解」にすると再現率の低さが隠れるため、候補に無い
    正解は `out_of_range` でも `self_loop` でもない扱いにせず、
    呼び出し側が候補集合と突き合わせられるよう件数だけを返す。
    """
    chosen = {(link.from_key, link.to_key) for link in result.predictions.links}
    for pair in correct:
        kind = result.kinds.get(pair)
        if kind is None:
            continue
        ok, total = counts.get(kind, (0, 0))
        counts[kind] = (ok + int(pair in chosen), total + 1)
    return counts


def missed_gt_transitions(
    correct: set[tuple[TrackKey, TrackKey]], result: DayResult
) -> list[tuple[TrackKey, TrackKey]]:
    """候補に入らなかった正解（種類が付けられないもの）。"""
    return [p for p in correct if p not in result.kinds]
