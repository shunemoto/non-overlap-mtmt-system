"""外観類似度がどれだけ正解と誤ペアを分けられているかの診断。

時空間側（`mct/st_report.py`）と同じ体裁にそろえてある。本フェーズの判断は
「外観が時空間より分けられているか」でしか下せないので、必ず時空間の
基準線を並べて表示する。

とくに自己ループは同一カメラ内の比較で、照明・画角・解像度がそろっている。
時空間は尤度が定数で埋まって順位が付かない条件なので、外観が効くなら
ここで最も高く出るはずである。逆にここで出なければ手法そのものを疑う。
"""

from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from mct.appearance import JOINT_NAMES, N_JOINTS, AppearanceDescriptor
from mct.appearance_sim import AppScore, sift_similarity
from mct.candidates import Candidate
from mct.metrics import auc, format_quantiles

__all__ = [
    "LabelledApp", "FIELDS", "print_separation", "print_self_loop_separation",
    "print_ranks", "print_joint_contribution", "print_missing_breakdown",
    "plot_separation",
]

PNG_NAME = "app_likelihood_separation.png"

# (表示名, 属性名)
FIELDS: tuple[tuple[str, str], ...] = (
    ("S_color", "s_color"),
    ("S_color_upper", "s_color_upper"),
    ("S_color_lower", "s_color_lower"),
    ("S_sift", "s_sift"),
    ("S_app", "s_app"),
)


@dataclass
class LabelledApp:
    """正解かどうかの印を付けた外観スコア。"""

    score: AppScore
    candidate: Candidate
    correct: bool
    date: str = ""

    @property
    def in_range(self) -> bool:
        return self.candidate.in_search_range and not self.candidate.is_self_loop

    @property
    def is_self_loop(self) -> bool:
        return self.candidate.is_self_loop


def _split(labelled: list[LabelledApp], attr: str) -> tuple[list[float], list[float]]:
    pos = [getattr(x.score, attr) for x in labelled if x.correct]
    neg = [getattr(x.score, attr) for x in labelled if not x.correct]
    return [v for v in pos if v is not None], [v for v in neg if v is not None]


def _auc_line(label: str, pos: list[float], neg: list[float], baseline: float | None) -> None:
    value = auc(pos, neg)
    note = ""
    if not np.isnan(value):
        if value < 0.5:
            note = "  ← 偶然以下"
        elif value < 0.6:
            note = "  ← ほぼ偶然"
        elif baseline is not None and not np.isnan(baseline):
            note = f"  ← 時空間 {baseline:.3f} を {value - baseline:+.3f}"
    print(f"  {label:<15} AUC {value:.3f}  (正解 {len(pos):>4} / 誤 {len(neg):>5}){note}")


def print_separation(
    labelled: list[LabelledApp],
    st_auc_all: float | None = None,
    st_auc_in_range: float | None = None,
) -> None:
    """5a。全候補と in_range 限定の両方で、各指標の AUC と分布を出す。"""
    print("\n" + "=" * 78)
    print("外観類似度の分離度（5a）")
    print("=" * 78)

    in_range = [x for x in labelled if x.in_range]
    for title, subset, baseline in (
        ("全候補", labelled, st_auc_all),
        ("in_range のみ（同一カメラの自己ループを除く）", in_range, st_auc_in_range),
    ):
        n_pos = sum(1 for x in subset if x.correct)
        print(f"\n[{title}] 候補 {len(subset)} 件（正解 {n_pos} / 誤 {len(subset)-n_pos}）")
        if baseline is not None and not np.isnan(baseline):
            print(f"  比較対象: 時空間尤度の AUC {baseline:.3f}")
        for label, attr in FIELDS:
            pos, neg = _split(subset, attr)
            _auc_line(label, pos, neg, baseline)

        print(f"\n  {'指標':<15}{'側':<5} p10/p25/中央/p75/p90")
        print("  " + "-" * 62)
        for label, attr in FIELDS:
            pos, neg = _split(subset, attr)
            print(f"  {label:<15}{'正解':<5} {format_quantiles(pos)}")
            print(f"  {'':<15}{'誤':<5} {format_quantiles(neg)}")


def print_self_loop_separation(
    labelled: list[LabelledApp], st_auc: float | None = None
) -> None:
    """5b。自己ループ候補だけで同じ集計をする。"""
    loops = [x for x in labelled if x.is_self_loop]
    n_pos = sum(1 for x in loops if x.correct)

    print("\n" + "=" * 78)
    print("自己ループの分離度（5b）")
    print("=" * 78)
    print(f"  自己ループ候補 {len(loops)} 件（正解 {n_pos} / 誤 {len(loops)-n_pos}）")
    if st_auc is not None and not np.isnan(st_auc):
        print(f"  比較対象: 時空間尤度の AUC {st_auc:.3f}（尤度が定数で順位が付かない）")
    if not loops or n_pos == 0:
        print("  対象がありません")
        return

    print("\n  同一カメラ内なので照明・画角・解像度がそろっており、"
          "外観が最も有利な条件にあたる")
    for label, attr in FIELDS:
        pos, neg = _split(loops, attr)
        _auc_line(label, pos, neg, st_auc)

    for label, attr in FIELDS:
        pos, neg = _split(loops, attr)
        print(f"\n  {label}  正解 {format_quantiles(pos)}")
        print(f"  {'':<{len(label)}}  誤   {format_quantiles(neg)}")


def _group_by_src(
    labelled: list[LabelledApp],
) -> dict[tuple[str, tuple[str, int]], list[LabelledApp]]:
    grouped: dict[tuple[str, tuple[str, int]], list[LabelledApp]] = collections.defaultdict(list)
    for x in labelled:
        grouped[(x.date, x.score.src)].append(x)
    return grouped


def _ranks(items: list[LabelledApp], target: LabelledApp, attr: str) -> tuple[float, float]:
    """(平均順位, 同値を最下位とした順位)。値が無いものは最下位側に置く。"""
    value = getattr(target.score, attr)
    if value is None:
        return float(len(items)), float(len(items))
    scores = [getattr(x.score, attr) for x in items]
    higher = sum(1 for v in scores if v is not None and v > value)
    tied = sum(1 for v in scores if v is not None and v == value)
    return higher + (tied + 1) / 2.0, float(higher + tied)


def print_ranks(
    labelled: list[LabelledApp],
    attr: str = "s_app",
    st_first_rate: dict[int, float] | None = None,
    scope: str = "in_range",
) -> dict[int, tuple[int, int]]:
    """5c。外観のみで順位付けしたときの、競合数別の1位率。

    分母の取り方は時空間側と合わせる。`in_range` だけ、あるいは自己ループ
    だけに絞らないと、種類の違う候補が混ざって順位が手法の性能を映さない。
    """
    by_src = _group_by_src(labelled)
    select: Callable[[LabelledApp], bool] = (
        (lambda x: x.is_self_loop) if scope == "self_loop" else (lambda x: x.in_range)
    )

    by_competitors: dict[int, list[bool]] = collections.defaultdict(list)
    ranks: list[tuple[float, float, int]] = []
    for items in by_src.values():
        subset = [x for x in items if select(x)]
        for target in (x for x in subset if x.correct):
            avg, worst = _ranks(subset, target, attr)
            ranks.append((avg, worst, len(subset)))
            by_competitors[min(len(subset), 4)].append(avg <= 1.0)

    label = "自己ループ" if scope == "self_loop" else "in_range"
    print(f"\n{label} の競合数ごとの1位率（{attr} の降順・同値は平均順位）")
    if not ranks:
        print("  対象がありません")
        return {}

    header = f"    {'候補数':<8}{'src数':>7}{'正解が1位':>10}{'1位率':>9}"
    if st_first_rate:
        header += f"{'時空間':>9}{'差':>9}"
    print(header)

    result: dict[int, tuple[int, int]] = {}
    for key in sorted(by_competitors):
        values = by_competitors[key]
        hits = sum(values)
        rate = 100 * hits / len(values)
        result[key] = (hits, len(values))
        line = f"    {'4以上' if key == 4 else str(key):<8}{len(values):>7}{hits:>10}{rate:>8.1f}%"
        if st_first_rate and key in st_first_rate:
            line += f"{st_first_rate[key]:>8.1f}%{rate - st_first_rate[key]:>+8.1f}"
        print(line)

    positions = [r for r, _, _ in ranks]
    print(f"\n  正解 {len(ranks)} 件 / 順位の中央値 {statistics.median(positions):.1f}"
          f" / 最悪 {max(positions):.1f}"
          f" / 分母の中央値 {int(statistics.median([n for _, _, n in ranks]))} 件")
    for name, index in (("平均順位", 0), ("同値を最下位とした順位", 1)):
        hits = sum(1 for r in ranks if r[index] <= 1.0)
        print(f"  {name:<22} 1位 {hits:>3} / {len(ranks)} = {100*hits/len(ranks):.1f}%")
    return result


def print_joint_contribution(
    labelled: list[LabelledApp],
    descriptors: dict[str, dict[tuple[str, int], AppearanceDescriptor]],
) -> None:
    """5f。6点それぞれを単独で使ったときの AUC。

    全点の平均に埋もれている弱い点を見つけるため、1点ずつ測り直す。
    `sift_similarity` に関節点を1つだけ渡して計算し直すので、記述子が
    要る（スコアの再計算ではなく、元の記述子から引き直す）。
    """
    print("\n" + "=" * 78)
    print("関節点ごとの寄与（5f。その点だけで S_sift を作った場合）")
    print("=" * 78)
    print(f"  {'関節点':<8}{'AUC':>8}{'正解':>7}{'誤':>8}{'両側で有効な割合':>18}")
    print("  " + "-" * 52)

    for j in range(N_JOINTS):
        pos, neg, usable = [], [], 0
        for x in labelled:
            table = descriptors.get(x.date) or {}
            a, b = table.get(x.score.src), table.get(x.score.tgt)
            if a is None or b is None:
                continue
            if not (a.sift_valid[j] and b.sift_valid[j]):
                continue
            usable += 1
            value = float(np.dot(a.sift[j], b.sift[j]))
            (pos if x.correct else neg).append(value)

        value = auc(pos, neg)
        ratio = 100 * usable / max(1, len(labelled))
        print(f"  {JOINT_NAMES[j]:<8}{value:>8.3f}{len(pos):>7}{len(neg):>8}{ratio:>17.1f}%")


def print_missing_breakdown(labelled: list[LabelledApp], n_skipped: int = 0) -> None:
    counts = collections.Counter(x.score.missing for x in labelled)
    total = max(1, len(labelled))
    print("\n欠損の内訳")
    labels = {
        None: "色・SIFT ともあり", "sift": "SIFT 欠損（色のみ）",
        "color": "色 欠損（SIFT のみ）", "both": "両方欠損（時空間のみ）",
    }
    for key, label in labels.items():
        n = counts.get(key, 0)
        print(f"  {label:<24}{n:>6} 件 ({100*n/total:.2f}%)")
    if n_skipped:
        print(f"  記述子が無く採点できなかった候補: {n_skipped} 件")

    joints = collections.Counter(x.score.n_joints for x in labelled)
    spread = " ".join(f"{k}点:{v}" for k, v in sorted(joints.items()))
    print(f"  両側で有効だった関節点の数: {spread}")


def plot_separation(labelled: list[LabelledApp], dest: str | Path) -> Path:
    """正解と誤ペアの外観類似度を重ねたヒストグラム。"""
    import matplotlib
    matplotlib.use("Agg")                      # 画面が無い環境で描く
    import matplotlib.pyplot as plt

    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        ("all candidates", labelled),
        ("in_range only", [x for x in labelled if x.in_range]),
        ("self loop only", [x for x in labelled if x.is_self_loop]),
    )
    fig, axes = plt.subplots(3, len(panels), figsize=(5 * len(panels), 11))
    bins = np.linspace(0, 1, 41)

    for col, (title, subset) in enumerate(panels):
        for row, (label, attr) in enumerate(
            (("S_color", "s_color"), ("S_sift", "s_sift"), ("S_app", "s_app"))
        ):
            ax = axes[row][col]
            pos, neg = _split(subset, attr)
            # 件数が桁で違うので割合で重ねる
            ax.hist(neg, bins=bins, density=True, alpha=0.55,
                    label=f"wrong (n={len(neg)})", color="#888888")
            ax.hist(pos, bins=bins, density=True, alpha=0.65,
                    label=f"correct (n={len(pos)})", color="#d1495b")
            ax.set_title(f"{title} / {label}   AUC={auc(pos, neg):.3f}")
            ax.set_xlabel(label)
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

    fig.suptitle("Appearance similarity separation")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
