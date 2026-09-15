"""Ground Truth との突き合わせによる評価。

予測トラックと GT トラックは ID が一致しないので、まず bbox の重なりで
1対1に対応付け、そのうえで Global ID の割り当てを評価する。

数えるのは**トラック単位**で、フレーム単位ではない。1トラックを1つの人物の
出現とみなし、それが正しい Global ID に統合されたかを見る。
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from mct.gt import GroundTruth, GTBox, GTTransition, gt_transitions
from mct.schema import FramePoint, TrackingResult

__all__ = [
    "TrackMatching",
    "EvalResult",
    "CandidateRecall",
    "match_tracks",
    "evaluate",
    "candidate_recall",
]


def _iou(a: Iterable[float], b: Iterable[float]) -> float:
    """[x, y, w, h] 同士の IoU。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackMatching:
    """1カメラ分の、予測トラックと GT トラックの対応。"""

    camera: str
    pairs: dict[str, int] = field(default_factory=dict)          # pred -> gt
    mean_iou: dict[str, float] = field(default_factory=dict)     # pred -> 平均 IoU
    overlap_ratio: dict[str, float] = field(default_factory=dict)
    unmatched_pred: list[str] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)
    ignored_pred: set[str] = field(default_factory=set)

    def evaluable_pairs(self, gt: GroundTruth) -> list[tuple[str, int]]:
        """評価対象の対（ignore の GT に対応付いたものを除く）。"""
        return [
            (p, g) for p, g in self.pairs.items()
            if (self.camera, g) not in gt.ignored
        ]


def _frame_boxes(points: list[FramePoint]) -> dict[int, list[float]]:
    return {p.frame_idx: list(p.bbox) for p in points}


def match_tracks(
    pred: TrackingResult,
    gt: GroundTruth,
    camera: str,
    params: dict[str, Any],
) -> TrackMatching:
    """予測トラックと GT トラックを1対1に対応付ける。

    重なりの強さは「共存フレームのうち IoU が閾値以上だったフレーム数」で測り、
    ハンガリー法で割り当てる。採用の可否は **GT トラック長に対する割合**で
    決める。GT は人物が明確に見える区間にしか付いていないので予測より短く、
    予測側を分母にすると正しい対応まで落ちてしまう。
    """
    iou_threshold = float(params.get("iou_threshold", 0.5))
    min_ratio = float(params.get("match_min_ratio", 0.3))

    matching = TrackMatching(camera=camera)
    gt_tracks = gt.tracks(camera)
    pred_ids = sorted(pred.tracks, key=lambda t: int(t))
    gt_ids = sorted(gt_tracks)

    if not pred_ids or not gt_ids:
        matching.unmatched_pred = list(pred_ids)
        matching.unmatched_gt = list(gt_ids)
        # GT が無ければ何も判定できないので、予測はすべて免責する
        matching.ignored_pred = set(pred_ids)
        return matching

    # GT がトラックを1つも持たないフレームは評価対象にしない
    gt_frames = gt.frames_with_gt(camera)
    pred_boxes = {t: _frame_boxes(pred.tracks[t]) for t in pred_ids}
    gt_boxes: dict[int, dict[int, list[float]]] = {
        t: {b.frame_idx: list(b.bbox) for b in gt_tracks[t]} for t in gt_ids
    }

    hits = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int32)
    iou_sums = np.zeros_like(hits, dtype=np.float64)
    for i, p in enumerate(pred_ids):
        boxes_p = pred_boxes[p]
        for j, g in enumerate(gt_ids):
            boxes_g = gt_boxes[g]
            shared = boxes_p.keys() & boxes_g.keys() & gt_frames
            for frame in shared:
                value = _iou(boxes_p[frame], boxes_g[frame])
                if value >= iou_threshold:
                    hits[i, j] += 1
                    iou_sums[i, j] += value

    rows, cols = linear_sum_assignment(-hits)
    matched_pred: set[str] = set()
    matched_gt: set[int] = set()
    for i, j in zip(rows, cols):
        n_hit = int(hits[i, j])
        if n_hit == 0:
            continue
        gt_len = len(gt_boxes[gt_ids[j]])
        ratio = n_hit / gt_len if gt_len else 0.0
        if ratio < min_ratio:
            continue
        pred_id, gt_id = pred_ids[i], gt_ids[j]
        matching.pairs[pred_id] = gt_id
        matching.mean_iou[pred_id] = float(iou_sums[i, j] / n_hit)
        matching.overlap_ratio[pred_id] = ratio
        matched_pred.add(pred_id)
        matched_gt.add(gt_id)

    matching.unmatched_pred = [t for t in pred_ids if t not in matched_pred]
    matching.unmatched_gt = [t for t in gt_ids if t not in matched_gt]

    # FP に数えない予測: ignore の GT に対応付いたものと、どこにも対応付かなかったもの
    matching.ignored_pred = set(matching.unmatched_pred) | {
        p for p, g in matching.pairs.items() if (camera, g) in gt.ignored
    }
    return matching


@dataclass
class EvalResult:
    date: str
    idtp: int = 0
    idfp: int = 0
    idfn: int = 0
    n_pred_evaluable: int = 0
    n_gt_evaluable: int = 0
    global_id_accuracy: float = 0.0
    n_gt_global_ids: int = 0
    n_gt_global_ids_correct: int = 0
    mean_ious: list[float] = field(default_factory=list)
    wrong_merges: list[dict[str, Any]] = field(default_factory=list)
    unmatched_pred: dict[str, int] = field(default_factory=dict)
    unmatched_gt: dict[str, int] = field(default_factory=dict)

    @property
    def idp(self) -> float:
        d = self.idtp + self.idfp
        return self.idtp / d if d else 0.0

    @property
    def idr(self) -> float:
        d = self.idtp + self.idfn
        return self.idtp / d if d else 0.0

    @property
    def idf1(self) -> float:
        d = 2 * self.idtp + self.idfp + self.idfn
        return 2 * self.idtp / d if d else 0.0


def evaluate(
    pred_global_ids: dict[str, dict[str, str]],
    gt: GroundTruth,
    matchings: dict[str, TrackMatching] | None = None,
    *,
    use_gt_tracks: bool = False,
) -> EvalResult:
    """Global ID の割り当てを評価する。

    `use_gt_tracks=True` は再同定単体の評価で、トラックの対応付けを飛ばして
    GT の track_id をそのまま予測の識別子として扱う。検出・追跡の誤りを
    含めずに再同定だけを測りたいときに使う。
    """
    result = EvalResult(date=gt.date)

    # (camera, gt_track, 予測 GID, GT GID) の並び
    entries: list[tuple[str, int, str | None, str]] = []
    for camera in gt.boxes:
        assigned = pred_global_ids.get(camera, {})
        if use_gt_tracks:
            for gt_track in gt.evaluable(camera):
                entries.append(
                    (camera, gt_track, assigned.get(str(gt_track)),
                     gt.global_ids[(camera, gt_track)])
                )
            result.unmatched_pred[camera] = 0
            result.unmatched_gt[camera] = 0
            continue

        matching = (matchings or {}).get(camera)
        if matching is None:
            continue
        for pred_track, gt_track in matching.evaluable_pairs(gt):
            entries.append(
                (camera, gt_track, assigned.get(pred_track),
                 gt.global_ids[(camera, gt_track)])
            )
            result.mean_ious.append(matching.mean_iou[pred_track])
        result.unmatched_pred[camera] = len(
            [t for t in matching.unmatched_pred if t not in matching.ignored_pred]
        )
        result.unmatched_gt[camera] = len(
            [t for t in matching.unmatched_gt if (camera, t) not in gt.ignored]
        )

    result.n_pred_evaluable = len(entries)
    result.n_gt_evaluable = sum(len(gt.evaluable(c)) for c in gt.boxes)

    assignment = _assign_global_ids(entries)
    result.idtp = sum(
        1 for _, _, pred_gid, gt_gid in entries
        if pred_gid is not None and assignment.get(pred_gid) == gt_gid
    )
    result.idfp = result.n_pred_evaluable - result.idtp
    result.idfn = result.n_gt_evaluable - result.idtp

    result.wrong_merges = [
        {
            "camera": camera, "gt_track": gt_track,
            "pred_global_id": pred_gid, "gt_global_id": gt_gid,
            "assigned_to": assignment.get(pred_gid),
        }
        for camera, gt_track, pred_gid, gt_gid in entries
        if pred_gid is None or assignment.get(pred_gid) != gt_gid
    ]
    _fill_global_id_accuracy(result, entries)
    return result


def _assign_global_ids(
    entries: list[tuple[str, int, str | None, str]]
) -> dict[str, str]:
    """予測 Global ID と GT Global ID をハンガリー法で1対1に割り当てる。

    同じトラックを指した回数を利得として最大化する。
    """
    counts: dict[tuple[str, str], int] = collections.Counter(
        (pred_gid, gt_gid) for _, _, pred_gid, gt_gid in entries if pred_gid is not None
    )
    if not counts:
        return {}

    pred_ids = sorted({p for p, _ in counts})
    gt_ids = sorted({g for _, g in counts})
    matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int32)
    for (p, g), n in counts.items():
        matrix[pred_ids.index(p), gt_ids.index(g)] = n

    rows, cols = linear_sum_assignment(-matrix)
    return {
        pred_ids[i]: gt_ids[j] for i, j in zip(rows, cols) if matrix[i, j] > 0
    }


def _fill_global_id_accuracy(
    result: EvalResult, entries: list[tuple[str, int, str | None, str]]
) -> None:
    """GT の Global ID のうち、過不足なく1つの予測 GID にまとまったものの割合。

    IDF1 はトラック単位なので一部だけ正しくても点が入るが、こちらは
    「人物ひとりを最後まで通してまとめられたか」を見る。

    条件は2つで、(1) その GT GID の全トラックが同一の予測 GID に入ること
    （分割していない）、(2) その予測 GID に他の GT GID のトラックが
    混ざっていないこと（併合していない）。(2) を入れないと、全員を1つの
    予測 GID にまとめた場合でも 100% になってしまい、指標の意味が失われる。
    """
    by_gt_gid: dict[str, set[str | None]] = collections.defaultdict(set)
    by_pred_gid: dict[str | None, set[str]] = collections.defaultdict(set)
    for _, _, pred_gid, gt_gid in entries:
        by_gt_gid[gt_gid].add(pred_gid)
        by_pred_gid[pred_gid].add(gt_gid)

    result.n_gt_global_ids = len(by_gt_gid)
    result.n_gt_global_ids_correct = sum(
        1
        for preds in by_gt_gid.values()
        if len(preds) == 1
        and None not in preds
        and len(by_pred_gid[next(iter(preds))]) == 1
    )
    result.global_id_accuracy = (
        result.n_gt_global_ids_correct / result.n_gt_global_ids
        if result.n_gt_global_ids else 0.0
    )


@dataclass
class CandidateRecall:
    n_gt: int = 0
    n_covered: int = 0
    missed: list[GTTransition] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.n_covered / self.n_gt if self.n_gt else 0.0


def candidate_recall(
    candidates: list[dict[str, Any]],
    gt: GroundTruth,
    matchings: dict[str, TrackMatching] | None = None,
) -> CandidateRecall:
    """GT の正解遷移のうち、候補として列挙された割合。

    候補は `{"from": {"cam", "track"}, "to": {"cam", "track"}}` の並び。
    向きは問わない（どちらの順で挙がっていても届いているとみなす）。

    候補は**予測側のトラックID**で書かれているので、`matchings` を渡して
    GT のトラックIDを予測側に翻訳してから照合する。渡さない場合は
    GT 空間どうしの比較になる（GT をそのまま候補にした動作確認用）。
    対応する予測トラックが無い遷移は、そもそも候補になり得ないので
    `missed` に入る。
    """
    listed: set[tuple[str, int, str, int]] = set()
    for c in candidates:
        try:
            a = (str(c["from"]["cam"]), int(c["from"]["track"]))
            b = (str(c["to"]["cam"]), int(c["to"]["track"]))
        except (KeyError, TypeError, ValueError):
            continue
        listed.add((*a, *b))
        listed.add((*b, *a))

    to_pred = _gt_to_pred_map(matchings) if matchings else None

    report = CandidateRecall()
    for transition in gt_transitions(gt):
        report.n_gt += 1
        ends = [
            (transition.from_cam, transition.from_track),
            (transition.to_cam, transition.to_track),
        ]
        if to_pred is not None:
            translated = [to_pred.get(e) for e in ends]
            if any(t is None for t in translated):
                report.missed.append(transition)
                continue
            ends = translated                       # type: ignore[assignment]

        if (*ends[0], *ends[1]) in listed:
            report.n_covered += 1
        else:
            report.missed.append(transition)
    return report


def _gt_to_pred_map(
    matchings: dict[str, TrackMatching]
) -> dict[tuple[str, int], tuple[str, int]]:
    """(camera, GT track) -> (camera, 予測 track) の対応。"""
    return {
        (camera, gt_track): (camera, int(pred_track))
        for camera, matching in matchings.items()
        for pred_track, gt_track in matching.pairs.items()
    }


# --------------------------------------------------------------------------
# 診断出力
# --------------------------------------------------------------------------

IOU_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def print_eval_result(result: EvalResult) -> None:
    print(f"\n--- {result.date} ---")
    print(
        f"  IDTP {result.idtp} / IDFP {result.idfp} / IDFN {result.idfn}"
        f"   (評価対象 予測 {result.n_pred_evaluable} / GT {result.n_gt_evaluable})"
    )
    print(
        f"  IDP {result.idp:.4f} / IDR {result.idr:.4f} / IDF1 {result.idf1:.4f}"
    )
    print(
        f"  Global ID 単位の正解率: {result.global_id_accuracy:.4f} "
        f"({result.n_gt_global_ids_correct}/{result.n_gt_global_ids})"
    )

    unmatched_p = {c: n for c, n in result.unmatched_pred.items() if n}
    unmatched_g = {c: n for c, n in result.unmatched_gt.items() if n}
    if unmatched_p or unmatched_g:
        print(f"  対応付かず: 予測 {unmatched_p or '{}'} / GT {unmatched_g or '{}'}")

    if result.wrong_merges:
        print(f"  誤った統合 {len(result.wrong_merges)} 件:")
        for m in result.wrong_merges[:10]:
            pred = m["pred_global_id"] or "(未割当)"
            print(
                f"    {m['camera']}/GT track {m['gt_track']}: "
                f"予測 {pred} -> {m['assigned_to'] or '(なし)'} / 正解 {m['gt_global_id']}"
            )
        if len(result.wrong_merges) > 10:
            print(f"    ... 他 {len(result.wrong_merges) - 10} 件")


def print_iou_distribution(results: list[EvalResult]) -> None:
    """対応付けの IoU 分布。

    閾値 0.5 のすぐ上に山があると、閾値を少し動かすだけで対応が崩れる。
    そうなっていないかを確かめるための表。
    """
    ious = np.array([v for r in results for v in r.mean_ious])
    if len(ious) == 0:
        print("\n対応付けの IoU: 対応付いた対がありません")
        return

    print(f"\n対応付けの IoU 分布（対応付いた {len(ious)} 対の平均 IoU）")
    for lo, hi in zip(IOU_BINS, IOU_BINS[1:]):
        n = int(((ious >= lo) & (ious < hi)).sum())
        bar = "#" * int(40 * n / len(ious))
        print(f"  {lo:.1f}-{min(hi,1.0):.1f}  {n:>5} ({100*n/len(ious):5.1f}%) {bar}")
    near = int(((ious >= 0.4) & (ious <= 0.6)).sum())
    print(f"  平均 {ious.mean():.3f} / 中央 {np.median(ious):.3f} / 最小 {ious.min():.3f}")
    print(
        f"  0.4〜0.6 に入る対: {near} 件 ({100*near/len(ious):.1f}%)"
        f"{'  ← 閾値に敏感' if near / len(ious) > 0.1 else '  ← 閾値の影響は小さい'}"
    )


def print_summary(results: list[EvalResult]) -> None:
    if not results:
        return
    print("\n" + "=" * 78)
    print("全日付の集計")
    print("=" * 78)
    head = f"{'date':<8}{'IDTP':>6}{'IDFP':>6}{'IDFN':>6}{'IDP':>9}{'IDR':>9}{'IDF1':>9}{'GID正解率':>11}"
    print(head)
    print("-" * len(head))
    for r in results:
        print(
            f"{r.date:<8}{r.idtp:>6}{r.idfp:>6}{r.idfn:>6}"
            f"{r.idp:>9.4f}{r.idr:>9.4f}{r.idf1:>9.4f}{r.global_id_accuracy:>11.4f}"
        )

    idtp = sum(r.idtp for r in results)
    idfp = sum(r.idfp for r in results)
    idfn = sum(r.idfn for r in results)
    idp = idtp / (idtp + idfp) if idtp + idfp else 0.0
    idr = idtp / (idtp + idfn) if idtp + idfn else 0.0
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0
    n_gid = sum(r.n_gt_global_ids for r in results)
    n_ok = sum(r.n_gt_global_ids_correct for r in results)
    print("-" * len(head))
    print(
        f"{'合計':<8}{idtp:>6}{idfp:>6}{idfn:>6}{idp:>9.4f}{idr:>9.4f}{idf1:>9.4f}"
        f"{(n_ok/n_gid if n_gid else 0):>11.4f}"
    )


def print_candidate_recall(date: str, report: CandidateRecall) -> None:
    print(f"\n--- {date} 候補の再現率 ---")
    print(f"  {report.n_covered} / {report.n_gt} = {report.recall:.4f}")
    if not report.missed:
        return

    by_pair = collections.Counter(
        (t.from_cam, t.to_cam) for t in report.missed
    )
    print(f"  漏れた遷移 {len(report.missed)} 件のカメラ対別内訳:")
    for (a, b), n in by_pair.most_common():
        label = f"{a} -> {b}" + ("（自己ループ）" if a == b else "")
        print(f"    {label:<28} {n:>4} 件")


# --------------------------------------------------------------------------
# 候補から落ちた遷移の分類
# --------------------------------------------------------------------------

MISS_REASONS = (
    "no_pred_track",     # 対応する予測トラックが無い（検出漏れ・フィルタで除去）
    "zone_none",         # どちらかの端点が遷移領域に該当しない
    "no_route",          # transitions.yaml にそのゾーン対の経路が無い
    "overlap",           # 出現区間が重なっている
    "below_min_time",    # 探索時間の下限を満たさない
    "other",
)


def classify_missed(
    missed: list[GTTransition],
    gt: GroundTruth,
    summaries: dict[tuple[str, int], Any],
    routes: list[Any],
    matchings: dict[str, TrackMatching] | None,
    fps: float,
) -> dict[str, list[GTTransition]]:
    """候補に入らなかった遷移を理由ごとに分ける。

    どこを直せば再現率が上がるかを切り分けるための分類。上流（検出・フィルタ）
    が原因なのか、ゾーン定義なのか、経路定義なのかで打ち手が変わる。
    `summaries` は (camera, 予測 track_id) をキーとする TrackSummary。
    """
    to_pred = _gt_to_pred_map(matchings) if matchings else None
    buckets: dict[str, list[GTTransition]] = {r: [] for r in MISS_REASONS}

    for t in missed:
        tgt_key, src_key = (t.from_cam, t.from_track), (t.to_cam, t.to_track)
        if to_pred is not None:
            tgt_key = to_pred.get(tgt_key)          # type: ignore[assignment]
            src_key = to_pred.get(src_key)          # type: ignore[assignment]
            if tgt_key is None or src_key is None:
                buckets["no_pred_track"].append(t)
                continue

        tgt = summaries.get(tgt_key)
        src = summaries.get(src_key)
        if tgt is None or src is None:
            buckets["no_pred_track"].append(t)
            continue
        if not tgt.exit_zones or not src.entry_zones:
            buckets["zone_none"].append(t)
            continue

        usable = [
            r for r in routes
            if r.from_cam == tgt.camera and r.from_zone in tgt.exit_zones
            and r.to_cam == src.camera and r.to_zone in src.entry_zones
        ]
        if not usable:
            buckets["no_route"].append(t)
        elif tgt.end_frame >= src.start_frame or (
            tgt.start_frame <= src.end_frame and src.start_frame <= tgt.end_frame
        ):
            buckets["overlap"].append(t)
        elif all(
            (src.start_frame - tgt.end_frame) < r.search_range_sec[0] * fps
            for r in usable
        ):
            buckets["below_min_time"].append(t)
        else:
            buckets["other"].append(t)
    return buckets


def print_missed_breakdown(buckets: dict[str, list[GTTransition]], n_gt: int) -> None:
    labels = {
        "no_pred_track": "対応する予測トラックが無い（検出漏れ・フィルタ除去）",
        "zone_none": "どちらかの端点が遷移領域に該当しない",
        "no_route": "transitions.yaml に経路が無い",
        "overlap": "出現区間が重なっている",
        "below_min_time": "探索時間の下限を満たさない",
        "other": "その他",
    }
    total = sum(len(v) for v in buckets.values())
    print(f"\n  候補に入らなかった {total} 件の内訳:")
    for reason, label in labels.items():
        items = buckets[reason]
        if not items:
            continue
        print(f"    {label}: {len(items)} 件 ({100*len(items)/n_gt:.1f}%)")
        by_pair = collections.Counter((t.from_cam, t.to_cam) for t in items)
        for (a, b), n in by_pair.most_common(4):
            loop = "（自己ループ）" if a == b else ""
            print(f"      {a} -> {b}{loop}: {n} 件")
        for t in items[:2]:
            print(f"      例: {t.from_cam}#{t.from_track} -> {t.to_cam}#{t.to_track} (GID {t.global_id})")
