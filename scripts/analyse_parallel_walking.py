"""並列歩行が時空間尤度の識別力を奪っているかを測る。

時空間尤度は「予測到着時刻と実際の出現時刻のずれ」で候補を順位付けする。
複数人が並んで歩いていると、どの候補も予測到着時刻がほぼ同じになり、
ずれの大小で差が付かなくなる。

そこで src トラックごとに、in_range 候補の**予測到着時刻のばらつき**
（最大 - 最小）を求め、それと「正解が1位になったか」の関係を見る。
ばらつきが小さい群で1位率が低ければ、並列歩行が原因である裏付けになる。

    python scripts/analyse_parallel_walking.py [outputs|outputs_gt|both]
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mct.cli import _correct_pairs_and_summaries  # noqa: E402
from mct.gt import gt_transitions  # noqa: E402
from mct.spatiotemporal import KIND_IN_RANGE  # noqa: E402
from mct.st_pipeline import load_candidates  # noqa: E402

DATES = ("0720", "0727", "0730", "0803")
FPS = 30.0

# ばらつきの区切り（フレーム）。60 フレーム = 2 秒、150 フレーム = 5 秒。
BUCKETS = ((0, 60, "60未満"), (60, 150, "60〜150"), (150, 10**9, "150以上"))

# 並列歩行とみなす退出時刻の近さ
PARALLEL_FRAMES = 60


@dataclass
class SrcGroup:
    """1つの src トラックに紐づく in_range 候補のまとまり。"""

    date: str
    src: tuple[str, int]
    spread: float             # 予測到着時刻の最大 - 最小（フレーム）
    n_candidates: int
    correct_rank: float | None    # 正解の平均順位。正解が候補に無ければ None


def _load_st(root: Path, date: str) -> dict:
    with (root / date / "st_scores.json").open(encoding="utf-8") as f:
        payload = json.load(f)
    return {
        ((s["tgt"][0], int(s["tgt"][1])), (s["src"][0], int(s["src"][1]))): s
        for s in payload["scores"]
    }


def collect(root: Path, use_gt: bool) -> tuple[list[SrcGroup], dict]:
    args = argparse.Namespace(
        out=str(root), params=str(REPO_ROOT / "configs" / "params.yaml"),
        gt_root=str(REPO_ROOT / "data" / "gt"), use_gt=use_gt,
    )
    groups: list[SrcGroup] = []
    parallel = {"n_transitions": 0, "n_parallel": 0, "examples": []}

    for date in DATES:
        summaries, candidates, _ = load_candidates(date, root)
        st = _load_st(root, date)
        correct = _correct_pairs_and_summaries(args, date, summaries)[0]
        by_key = {s.key: s for s in summaries}

        by_src: dict[tuple[str, int], list] = collections.defaultdict(list)
        for candidate in candidates:
            row = st.get((candidate.tgt, candidate.src))
            if row is None or row["kind"] != KIND_IN_RANGE:
                continue
            if row.get("predicted_arrival") is None:
                continue                      # 到着時刻を予測できていない
            by_src[candidate.src].append((candidate, row))

        for src, items in by_src.items():
            arrivals = [row["predicted_arrival"] for _, row in items]
            spread = max(arrivals) - min(arrivals)

            target = next(
                (row for c, row in items if (c.tgt, c.src) in correct), None
            )
            rank = None
            if target is not None:
                values = [row["likelihood"] for _, row in items]
                higher = sum(1 for v in values if v > target["likelihood"])
                tied = sum(1 for v in values if v == target["likelihood"])
                rank = higher + (tied + 1) / 2.0

            groups.append(SrcGroup(date, src, spread, len(items), rank))

        parallel = _count_parallel(date, args, by_key, parallel)
    return groups, parallel


def _count_parallel(date, args, by_key, acc):
    """GT の正解遷移のうち、tgt の退出時刻が他トラックと近いものを数える。

    「他の候補」ではなく「同じカメラの他のトラック」を見る。並列歩行は
    候補の作られ方に関係なく起きる現象で、その頻度を知りたいため。
    """
    import mct.cli as cli

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = cli._load_gt(args, date)

    ends_by_cam: dict[str, list[tuple[int, tuple[str, int]]]] = collections.defaultdict(list)
    for key, s in by_key.items():
        ends_by_cam[key[0]].append((s.end_frame, key))

    correct, _ = _correct_pairs_and_summaries(args, date, list(by_key.values()))
    for tgt, _src in correct:
        s = by_key.get(tgt)
        if s is None:
            continue
        acc["n_transitions"] += 1
        near = [
            k for end, k in ends_by_cam[tgt[0]]
            if k != tgt and abs(end - s.end_frame) <= PARALLEL_FRAMES
        ]
        if near:
            acc["n_parallel"] += 1
            if len(acc["examples"]) < 6:
                acc["examples"].append(
                    f"{date} {tgt[0]}#{tgt[1]}（退出 {s.end_frame}）と "
                    + ", ".join(f"#{k[1]}" for k in near[:3])
                )
    return acc


def print_report(label: str, groups: list[SrcGroup], parallel: dict) -> None:
    print("\n" + "=" * 86)
    print(f"並列歩行の影響 — {label}")
    print("=" * 86)

    contested = [g for g in groups if g.n_candidates >= 2 and g.correct_rank is not None]
    print(f"  in_range 候補が2件以上あり、正解も候補にある src: {len(contested)} 件")
    print(f"  （全 src グループ {len(groups)} 件のうち）")

    print(f"\n  {'予測到着時刻のばらつき':<22}{'src数':>7}{'正解が1位':>10}{'1位率':>9}"
          f"{'候補数の中央':>13}{'ばらつき中央':>13}")
    print("  " + "-" * 76)
    for lo, hi, name in BUCKETS:
        sel = [g for g in contested if lo <= g.spread < hi]
        if not sel:
            print(f"  {name + ' フレーム':<22}{'（該当なし）':>7}")
            continue
        first = sum(1 for g in sel if g.correct_rank <= 1.0)
        print(
            f"  {name + ' フレーム':<22}{len(sel):>7}{first:>10}"
            f"{100*first/len(sel):>8.1f}%"
            f"{int(statistics.median([g.n_candidates for g in sel])):>13}"
            f"{statistics.median([g.spread for g in sel]):>12.0f}f"
        )

    def rate(items):
        return 100 * sum(1 for g in items if g.correct_rank <= 1.0) / len(items)

    small = [g for g in contested if g.spread < 60]
    large = [g for g in contested if g.spread >= 150]
    if small and large:
        print(f"\n  ばらつき 60未満 {rate(small):.1f}% vs 150以上 {rate(large):.1f}%"
              f"  （差 {rate(small) - rate(large):+.1f} ポイント）")
        print("    ※ この比較はそのままでは読めない。ばらつきが小さい src は候補数も"
              "\n      少ない傾向があり（中央値 2 対 3）、候補が少なければ当てずっぽうでも"
              "\n      1位になりやすいので、2つの効果が逆向きに混ざる。")

    # ばらつきと候補数は交絡するので、候補数をそろえて比べ直す
    print(f"\n  候補数をそろえた比較（in_range 候補がちょうど2件の src のみ）")
    two = [g for g in contested if g.n_candidates == 2]
    print(f"  {'ばらつき':<22}{'src数':>7}{'正解が1位':>10}{'1位率':>9}")
    print("  " + "-" * 50)
    for lo, hi, name in BUCKETS:
        sel = [g for g in two if lo <= g.spread < hi]
        if not sel:
            continue
        print(f"  {name + ' フレーム':<22}{len(sel):>7}"
              f"{sum(1 for g in sel if g.correct_rank <= 1.0):>10}{rate(sel):>8.1f}%")

    two_small = [g for g in two if g.spread < 60]
    two_large = [g for g in two if g.spread >= 150]
    if two_small and two_large:
        diff = rate(two_small) - rate(two_large)
        print(f"\n  候補数2件にそろえると 60未満 {rate(two_small):.1f}%"
              f"（{len(two_small)}件） vs 150以上 {rate(two_large):.1f}%"
              f"（{len(two_large)}件）  差 {diff:+.1f} ポイント")
        if diff < 0:
            print("    ばらつきが小さい群ほど1位率が低い。候補どうしの予測到着時刻が"
                  "\n    近接していると時空間尤度で差が付かない、という説明と整合する。"
                  "\n    ただし 150以上の群は件数が少ないので、幅は参考値にとどまる。")
        else:
            print("    候補数をそろえても、ばらつきが小さい群の方が1位率が高い。"
                  "\n    並列歩行だけでは説明できない。")

    n = parallel["n_transitions"]
    print(f"\n  並列歩行の頻度（GT の正解遷移のうち、退出側が同じカメラの別トラックと"
          f"\n  {PARALLEL_FRAMES} フレーム以内に退出しているもの）")
    print(f"    {parallel['n_parallel']} / {n} 件 = {100*parallel['n_parallel']/max(1,n):.1f}%")
    for example in parallel["examples"][:4]:
        print(f"      例: {example}")


def main(argv: list[str]) -> int:
    which = argv[1] if len(argv) > 1 else "both"
    for label, root, use_gt in (
        ("予測側 (outputs)", REPO_ROOT / "outputs", False),
        ("GT 側 (outputs_gt)", REPO_ROOT / "outputs_gt", True),
    ):
        if which not in ("both", root.name):
            continue
        groups, parallel = collect(root, use_gt)
        print_report(label, groups, parallel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
