"""crop / mask を保存するフレームの選抜。

再同定に使うのは人物の胴体まわりの外観なので、肩と腰がはっきり見えている
フレームほど価値が高い。一方で同じ瞬間ばかり選ぶと見えの多様性が失われるため、
トラックを時間で等分し、各区画から骨格信頼度の高いものを取る。

追跡モデルに依存しないので、重い import なしで単体テストできる。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["SCORE_KEYPOINTS", "pose_score", "select_frames"]

# COCO 17点のインデックス: 5=左肩, 6=右肩, 11=左腰, 12=右腰
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
SCORE_KEYPOINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)


def pose_score(kps: np.ndarray) -> float:
    """スコアに使う6点の信頼度の合計を返す。

    `kps` は COCO 17点の (17, 3) 配列で、各行が [x, y, conf]。

    6点の内訳は、両肩・両腰の実測4点に加えて、そこから算出する2点
    （両肩の中点と、両肩・両腰の中点）。算出点は実測値を持たないので、
    信頼度は元になった点の平均とする。
    """
    c_ls = float(kps[L_SHOULDER, 2])
    c_rs = float(kps[R_SHOULDER, 2])
    c_lh = float(kps[L_HIP, 2])
    c_rh = float(kps[R_HIP, 2])

    shoulder_mid = (c_ls + c_rs) / 2.0
    torso_mid = (c_ls + c_rs + c_lh + c_rh) / 4.0

    return c_ls + c_rs + c_lh + c_rh + shoulder_mid + torso_mid


def select_frames(
    scores: Sequence[float], n_target: int = 90, n_bins: int = 9
) -> list[int]:
    """保存対象フレームのインデックスを時間順に返す。

    `scores` はトラックのフレーム列と同じ長さ・同じ順序の `pose_score` の列。
    返るのは `scores` へのインデックスで、フレーム番号そのものではない。

    トラック長が `n_target` 以下なら全フレームを選ぶ。そうでなければ、
    時間順に `n_bins` 等分した各区画からスコア上位 `ceil(n_target/n_bins)` 件を
    取り、`n_target` に足りない分を全体のスコア上位から補充する。
    同点は先に現れたフレームを優先し、結果が実行ごとに揺れないようにする。
    """
    n = len(scores)
    if n_target <= 0 or n_bins <= 0:
        raise ValueError("n_target and n_bins must be positive")
    if n <= n_target:
        return list(range(n))

    # 同点時に先頭側を優先するための並べ替えキー
    def by_score_desc(indices: Sequence[int]) -> list[int]:
        return sorted(indices, key=lambda i: (-scores[i], i))

    per_bin = math.ceil(n_target / n_bins)
    chosen: set[int] = set()
    for bin_indices in np.array_split(np.arange(n), n_bins):
        chosen.update(by_score_desc(bin_indices.tolist())[:per_bin])

    # 区画の要素数が per_bin に満たない場合などで不足する分を全体上位から補う
    if len(chosen) < n_target:
        for i in by_score_desc(range(n)):
            if len(chosen) >= n_target:
                break
            chosen.add(i)

    # n_bins が n_target を割り切らないと per_bin の切り上げで超過し得る
    if len(chosen) > n_target:
        chosen = set(by_score_desc(sorted(chosen))[:n_target])

    return sorted(chosen)
