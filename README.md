# non-overlap-mtmt-system

視野が重複しない8台のカメラで撮影した映像から、カメラ間で同一人物を
再同定（Multi-Target Multi-Camera Tracking）する研究システム。

## 実装状況

**段階1（現在）: データ構造とリポジトリの骨組みのみ。**
追跡・再同定のアルゴリズムはまだ実装されていない。
CLI のサブコマンドは定義済みだが、呼び出すと `NotImplementedError` を送出する。

## ディレクトリ構成

```
mct/
  schema.py          追跡結果のデータ構造（FramePoint / TrackMeta / TrackingResult）
  io_utils.py        JSON の保存・読み込みと出力パス規約
  cli.py             コマンドラインエントリポイント
configs/
  cameras.yaml       カメラ定義（テンプレート）
  transitions.yaml   カメラ間の移動時間（テンプレート）
  params.yaml        追跡パラメータの既定値
scripts/             実験用スクリプト置き場
tests/
data/                入力映像（git 管理外）
outputs/             追跡結果・crop・mask（git 管理外）
```

## セットアップ

Python 3.11 を想定。

```bash
python -m pip install -e ".[dev]"
```

### torch は別途インストールする

`pyproject.toml` には `torch` を含めていない。PyPI の既定の torch は
環境によって CPU 版になるため、**CUDA 版を先に入れてから**上記を実行すること。

```bash
# 例: CUDA 12.1 環境
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

自分の環境に合うコマンドは <https://pytorch.org/get-started/locally/> で確認する。
インストール後、`python -c "import torch; print(torch.cuda.is_available())"` が
`True` を返すことを確認する。

## 使い方

```bash
# 1カメラ分の映像を追跡して outputs/ に結果を保存する
python -m mct.cli track --video data/0720/camA.mp4 --date 0720 --camera camA

# 追跡結果を重畳した動画を書き出す
python -m mct.cli viz-video --date 0720 --camera camA

# 足元点の軌跡図を書き出す
python -m mct.cli viz-trajectory --date 0720 --camera camA
```

共通オプション: `--out-root`（既定 `outputs`）、`--params`（既定 `configs/params.yaml`）。

## 出力

`outputs/{date}/{camera}/` 以下に次を出力する。

| パス | 内容 |
| --- | --- |
| `tracking_result.json` | 追跡結果（下記のデータ形式） |
| `crops/t{track_id:03d}_f{frame_idx:06d}.jpg` | 人物の切り出し画像 |
| `masks/t{track_id:03d}_f{frame_idx:06d}.png` | セグメンテーションマスク |

crop / mask はトラックごとに最大 `n_saved_per_track` フレーム分だけ保存する。
保存したフレームは `FramePoint.saved = True` になる。

## データ形式

`tracking_result.json` はファイルサイズを抑えるためインデントなしで書き出す
（非 ASCII 文字はエスケープしない）。構造は次のとおり。

```jsonc
{
  "meta": {
    "date": "0720",            // 撮影日
    "camera": "camA",          // カメラID
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "n_frames": 1800,
    "model_pose": "yolo11x-pose.pt",
    "model_seg": "yolo11x-seg.pt",
    "tracker": "botsort.yaml",
    "n_saved_per_track": 90,
    "created_at": "2026-07-20T13:45:00+09:00"  // ISO8601
  },
  "tracks": {
    "1": [                     // key = local_track_id（カメラ内で閉じたID）
      {
        "frame_idx": 1,        // 1-based（MOT 形式に合わせる）
        "bbox": [100.0, 200.0, 50.0, 120.0],  // [x, y, w, h] 左上原点
        "conf": 0.91,
        "foot_point": [125.0, 320.0],         // [x, y] bbox 下辺中央
        "saved": true,         // crop/mask を保存したフレームか
        "keypoints": [[0.0, 0.0, 0.5], "..."] // COCO 17点 [[x, y, conf], ...]
      }
    ]
  }
}
```

`keypoints` は `saved: true` のフレームにのみ格納し、`saved: false` のときは
`null` になる。`local_track_id` はカメラ内でのみ一意で、カメラをまたいだ
同一性は段階2以降の再同定で付与する。

Python からは `mct.schema` / `mct.io_utils` を使う。

```python
from mct import load_tracking_result, output_dir, crop_path

result = load_tracking_result("outputs/0720/camA/tracking_result.json")
for track_id, points in result.tracks.items():
    saved = [p for p in points if p.saved]
    print(track_id, len(points), "frames,", len(saved), "saved")
```

## テスト

```bash
pytest
```
