# non-overlap-mtmt-system

視野が重複しない8台のカメラで撮影した映像から、カメラ間で同一人物を
再同定（Multi-Target Multi-Camera Tracking）する研究システム。

## 実装状況

**フェーズ1: 1カメラ分の追跡と特徴保存まで。**
カメラ間の再同定はまだ実装されていない。観測区間・遷移領域も未定義のため、
検出された全トラックをフィルタせずに保存する。

## ディレクトリ構成

```
mct/
  schema.py          追跡結果のデータ構造（FramePoint / TrackMeta / TrackingResult）
  io_utils.py        JSON の保存・読み込みと出力パス規約
  config.py          params.yaml の読み込み
  select.py          保存フレームの選抜（追跡モデルに非依存）
  track.py           追跡本体（2パス構成）
  viz.py             可視化4種（tracks.json だけを読む）
  cli.py             コマンドラインエントリポイント
configs/
  cameras.yaml       カメラ定義（テンプレート）
  transitions.yaml   カメラ間の移動時間（テンプレート）
  params.yaml        追跡パラメータの既定値
scripts/
  link_videos.py     元データへの symlink を data/videos/ に張る
  bench.py           処理速度の計測
  run_viz_all.sh     4種類の可視化をまとめて生成する
tests/
data/videos/         入力映像への symlink（git 管理外）
outputs/             追跡結果・crop・mask（git 管理外）
```

## セットアップ

Python 3.11 以上。**ネイティブ arm64 のインタプリタを使うこと。**
anaconda の Python は x86_64 ビルドで Rosetta 動作になり、MPS の性能が出ない。

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision   # ← 先に入れる（下記参照）
python -m pip install -e ".[dev]"
```

### torch は別途インストールする

`pyproject.toml` には `torch` を含めていない。プラットフォームごとに
必要なビルドが違うためで、環境に合ったものを先に入れる。

- **Apple Silicon (M1/M2/...)**: PyPI の既定ホイールがそのまま Metal (MPS) に
  対応しているので `pip install torch torchvision` でよい。CUDA は使えない。
- **NVIDIA GPU 機**: <https://pytorch.org/get-started/locally/> の指示に従って
  CUDA 版を入れ、`configs/params.yaml` の `device` を `"cuda"` にする。

確認:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"   # Apple Silicon
```

MPS には未実装の演算が残っているため、`mct.track` は import 時に
`PYTORCH_ENABLE_MPS_FALLBACK=1` を設定し、未対応演算を CPU に落とす。

### 入力動画

`scripts/link_videos.py` が、元データ（日付ごとのフォルダに
`0720_cameraA.mp4` の形式で置かれている）への symlink を
`data/videos/{date}/{camera}.mp4` として張る。26 GB を複製せずに済む。

```bash
python scripts/link_videos.py            # 既定: ~/Downloads/情報シンポ2026/シナリオ
python scripts/link_videos.py /path/to/src
```

## 使い方

```bash
# 1カメラ分の映像を追跡して outputs/ に結果を保存する
python -m mct.cli track --date 0720 --camera camA

# 可視化（4種類をまとめて生成）
scripts/run_viz_all.sh 0720 camA

# 個別に実行することもできる
python -m mct.cli viz-video      --date 0720 --camera camA
python -m mct.cli viz-trajectory --date 0720 --camera camA
python -m mct.cli viz-density    --date 0720 --camera camA
python -m mct.cli viz-endpoints  --date 0720 --camera camA
```

### 可視化の使い分け

観測区間と遷移領域はこれらの図を見て決めるので、目的ごとに図を分けている。
いずれも `tracks.json` だけを読むので、追跡を再実行せずに描き直せる。

| コマンド | 出力 | 何を読むための図か |
| --- | --- | --- |
| `viz-video` | `{camera}_track.mp4` | 追跡が正しく追えているかをフレーム単位で確認する |
| `viz-trajectory` | `{camera}_trajectory.png` | 動線の全体像をつかむ |
| `viz-density` | `{camera}_density.png` | 検出が安定している範囲と、端で急に途切れる位置を見る |
| `viz-endpoints` | `{camera}_endpoints.png` | 遷移領域の矩形と、進入・退出方向の角度範囲を決める |

`viz-video` は**人物が1人も検出されていないフレームを出力しない**。撮影時間の
大半が無人だと流して見るだけで時間を取られるため。フレームが飛ぶので元動画の
フレーム番号を画面に焼き込む。足元点が `--jump-threshold`（既定150px）以上
跳んだフレームは bbox を赤枠にして `JUMP` と表示する。遮蔽で bbox の下辺が
切り上がった箇所を見つけるための目印で、判定は真に連続する2フレームの間に
限る（トラックには最大20フレーム程度の欠落があり、間が空けば足元点が離れるのは
当然のため）。

`viz-endpoints` では、トラック長が30フレーム未満のものを灰色の小さな点にして
区別する。一瞬の写り込みや誤検出を遷移領域の判断材料から外せるようにするため。
座標を読み取れるよう200px間隔のグリッドと目盛りを重ねてある。

`--video` を省くと `data/videos/{date}/{camera}.mp4` を使う。
その他の主なオプション:

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `--out` | `outputs` | 出力ルート |
| `--config` | `configs/params.yaml` | パラメータ YAML |
| `--device` | YAML の値 | `mps` / `cuda` / `cpu` |
| `--stride` | 1 | N フレームに1枚だけ推論する |
| `--model-pose`, `--model-seg` | YAML の値 | モデルの上書き |
| `--max-frames` | なし | 先頭 N フレームで打ち切る（動作確認用） |
| `--jump-threshold` | 150 | `viz-video` で JUMP とみなす足元点の移動量 [px] |
| `--bin-px` | 20 | `viz-density` のヒストグラムのビン幅 [px] |

## 処理の流れ

追跡は2パス構成。全フレームに seg をかけると時間もディスクも見合わないため、
保存するフレームを絞ってから2パス目を回す。

1. **1パス目 (pose)**: `model.track(stream=True)` で全フレームを逐次処理し、
   bbox・conf・足元点・骨格点を得る。骨格点はメモリ上に保持し、最終的な
   JSON には選抜フレーム分しか書かない。
2. **選抜**: トラックを時間で9等分し、各区画から「肩・腰まわり6点の信頼度の
   合計」が高い上位10枚、計90枚を選ぶ。足りなければ全体の上位から補う。
   トラック長が90未満なら全フレーム。
3. **2パス目 (seg)**: 選抜されたフレームだけに seg を適用する。動画は先頭から
   順次読む1パスのみで、同じフレームに複数トラックがあっても推論は1回。
   各トラックの bbox と最も IoU が高いマスクを採用し、IoU が 0.3 未満なら
   マスク無しとして記録する。

### 処理速度（Apple M1 / 8-core GPU / 16 GB, imgsz=640）

| モデル | 1パス目 | 12,902フレームあたり |
| --- | --- | --- |
| `yolo11x-pose` | 7.4 fps | 29.0 分 |
| `yolo11l-pose` | 14.6 fps | 14.7 分 |
| `yolo11x-seg`（2パス目） | 6.3 fps | 選抜枚数に比例 |

32本を現実的な時間で回すため、既定は `yolo11l-pose`（全32本で約10.7時間）。
精度優先なら `configs/params.yaml` の `model_pose` を `yolo11x-pose.pt` に
変える（約18.3時間）。

## 出力

`outputs/{date}/{camera}/` 以下に次を出力する。

| パス | 内容 |
| --- | --- |
| `tracks.json` | 追跡結果（下記のデータ形式） |
| `crops/t{track_id:03d}_f{frame_idx:06d}.jpg` | 人物の切り出し画像 |
| `masks/t{track_id:03d}_f{frame_idx:06d}.png` | セグメンテーションマスク |
| `viz/{camera}_track.mp4` | `viz-video` の出力 |
| `viz/{camera}_trajectory.png` | `viz-trajectory` の出力 |
| `viz/{camera}_density.png` | `viz-density` の出力 |
| `viz/{camera}_endpoints.png` | `viz-endpoints` の出力 |

crop はトラックごとに最大 `n_saved_per_track` 枚。マスクは crop と同一サイズの
2値 PNG（人物=255, 背景=0）で、IoU が閾値に満たなかったフレームには作られない。

## データ形式

`tracks.json` はファイルサイズを抑えるためインデントなしで書き出す
（非 ASCII 文字はエスケープしない）。構造は次のとおり。

```jsonc
{
  "meta": {
    "date": "0720",            // 撮影日
    "camera": "camA",          // カメラID
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "n_frames": 12902,
    "model_pose": "yolo11l-pose.pt",
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
        "saved": true,         // crop を保存したフレームか
        "keypoints": [[0.0, 0.0, 0.5], "..."],// COCO 17点 [[x, y, conf], ...]
        "crop_box": [95.0, 188.0, 60.0, 144.0]// 実際に切り出した矩形 [x,y,w,h]
      }
    ]
  }
}
```

`keypoints` と `crop_box` は `saved: true` のフレームにのみ格納し、
`saved: false` では `null` になる。座標はいずれも元画像座標系なので、
骨格点を crop 画像の座標系に移すには `crop_box` の `x`, `y` を引く。
`crop_box` は bbox を `crop_padding_ratio` だけ広げ、画像端でクリップした
実矩形なので bbox とは一致しない。

`local_track_id` はカメラ内でのみ一意で、カメラをまたいだ同一性は
今後の再同定フェーズで付与する。

Python からは `mct.schema` / `mct.io_utils` を使う。

```python
from mct import load_tracking_result

result = load_tracking_result("outputs/0720/camA/tracks.json")
for track_id, points in result.tracks.items():
    saved = [p for p in points if p.saved]
    print(track_id, len(points), "frames,", len(saved), "saved")
```

## テスト

```bash
pytest
```

`schema` と `select` のテストは torch や ultralytics を import しないので、
モデルを持たない環境でも動く。
