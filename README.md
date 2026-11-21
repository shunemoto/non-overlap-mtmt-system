# non-overlap-mtmt-system

視野が重複しない8台のカメラで撮影した映像から、カメラ間で同一人物を
再同定（Multi-Target Multi-Camera Tracking）する研究システム。

## システムの概要

1. `track` で YOLO11 と BoT-SORT を使い、カメラごとに人物を追跡する。
2. `filter` で観測区間を適用し、短いトラックや静止誤検出を除く。
3. `candidates` で遷移領域、進行方向、カメラ間の接続関係から照合候補を作る。
4. `st-scores` で移動時間などから時空間尤度を求め、`appearance` と `app-scores` で色ヒストグラムと SIFT による外観類似度を求める。
5. `reid` でスコアを融合してトラックを対応付け、撮影日ごとの Global ID を割り当てる。正解データ（GT）と比較して IDF1 などを評価できる。

## ディレクトリ構成

```
mct/
  schema.py          追跡結果のデータ構造（FramePoint / TrackMeta / TrackingResult）
  io_utils.py        JSON の保存・読み込みと出力パス規約
  config.py          params.yaml の読み込み
  geometry.py        領域・角度の判定（追跡モデルに非依存）
  select.py          保存フレームの選抜（追跡モデルに非依存）
  track.py           追跡本体（2パス構成）
  candidate_pipeline.py  トラックの要約と遷移候補の生成
  st_pipeline.py     時空間スコアの入出力
  appearance.py      外観記述子の生成
  reid_pipeline.py   スコアの融合、割り当て、Global ID の統合
  evaluation.py      正解データとの比較
  viz.py             可視化5種（追跡モデルを import しない）
  cli.py             コマンドラインエントリポイント
configs/
  cameras.yaml       観測区間と遷移領域の定義
  transitions.yaml   カメラ間の移動時間（テンプレート）
  transitions_calibrated.yaml  距離を校正済みの遷移設定（既定）
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

Python 3.11 以上。以下は Apple Silicon での例。MPS を使う場合は
ネイティブ arm64 の Python を使う。

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
  CUDA 版を入れ、`configs/params.yaml` の `tracking.device` を `"cuda"` にする。
- **CPU**: `configs/params.yaml` の `tracking.device` を `"cpu"` にする。

確認:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"   # Apple Silicon
```

MPS には未実装の演算が残っているため、`mct.track` は import 時に
`PYTORCH_ENABLE_MPS_FALLBACK=1` を設定し、未対応演算を CPU に落とす。

### 入力動画と正解データ

撮影動画はクラウドで個別共有する。受け取った動画は
`data/videos/{date}/{camera}.mp4`（例 `data/videos/0720/camA.mp4`）に配置する。
元データが `<元データのルート>/<日付>/*_時刻同期/0720_cameraA.mp4` のような構成なら、
ファイルを複製せずにシンボリックリンクで配置できる。

```bash
python scripts/link_videos.py /path/to/source
```

正解データの共有先は [non-overlap-mtmt-dataset](https://github.com/shunemoto/non-overlap-mtmt-dataset)。

| ファイル | このシステムでの配置先 |
| --- | --- |
| カメラごとの MOT アノテーション | `data/gt/{date}/{camera}/gt.txt` |
| Global ID 対応表（データセット側に追加予定） | `data/gt/globalID.csv` |

対応表の列は `sequence_id,global_id,camera_id,local_track_id`。
モデル重み、入力データ、生成した動画や追跡結果は Git の管理対象外。

## 使い方

### 1カメラで動作を確認する

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

### 研究データで再同定と評価を実行する

全4撮影日分の動画と正解データを配置してから、リポジトリのルートで実行する。
時空間スコアの計算は既定で4日分の正解遷移から到着時刻の誤差を推定する。
同じ手順で特定の日をその推定から外すには、`st-scores` に `--holdout-date 0720` のように指定する。

```bash
for date in 0720 0727 0730 0803; do
  for camera in camA camB camC camD camE camF camG camH; do
    python -m mct.cli track --date "$date" --camera "$camera" || exit 1
  done
done

python -m mct.cli filter --all-dates
python -m mct.cli candidates --all-dates
python -m mct.cli st-scores --all-dates
python -m mct.cli appearance --all-dates
python -m mct.cli app-scores --all-dates
python -m mct.cli reid --all-dates
```

`reid` は既定で手法6を実行し、`outputs/{date}/global_ids_m6.json` に結果を書き出して、
正解データとの比較結果を表示する。対応表の入手前でも、単一カメラの追跡と可視化を試せる。

### 可視化の使い分け

観測区間と遷移領域はこれらの図を見て決めるので、目的ごとに図を分けている。
追跡を再実行せずに描き直せる。`viz-video` と `viz-region` は元動画も使う。

| コマンド | 出力 | 何を読むための図か |
| --- | --- | --- |
| `viz-video` | `{camera}_track.mp4` | 追跡が正しく追えているかをフレーム単位で確認する |
| `viz-trajectory` | `{camera}_trajectory.png` | 動線の全体像をつかむ |
| `viz-density` | `{camera}_density.png` | 検出が安定している範囲と、端で急に途切れる位置を見る |
| `viz-endpoints` | `{camera}_endpoints.png` | 遷移領域の矩形と、進入・退出方向の角度範囲を決める |
| `viz-region` | `{camera}_region.png` | 定義した観測区間・遷移領域が意図どおりか確かめる |

`viz-region` だけは `tracks.json` を読まず、**動画と `configs/cameras.yaml` だけ**で
動く。領域の座標を書いては確認する作業を、追跡の実行中にも並行してできる。

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

## 観測区間と遷移領域の定義

`configs/cameras.yaml` にカメラごとの領域を書き、`viz-region` で確認する。

```bash
python -m mct.cli viz-region --date 0720 --camera camA
```

```yaml
cameras:
  camA:
    image_size: [1920, 1080]
    exclude_regions:            # 観測区間の「外側」。ここに入る点は後段で削除する
      - type: rect
        points: [[0, 0], [1920, 180]]          # 左上, 右下
      - type: polygon
        points: [[0, 900], [420, 1080], [0, 1080]]
    transition_zones:
      - id: a_corridor
        region:
          type: rect
          points: [[820, 300], [1500, 1080]]
        exit_dir: [290, 60]     # 退出時の進行方向の許容範囲（度）
        entry_dir: [110, 240]   # 出現時の進行方向の許容範囲（度）
        to:
          - {cam: camB, room: null}
          - {cam: camA, room: design_room}     # 自己ループ
    homography:                 # 画像座標 → BEV 実空間座標。未計測なら空でよい
      image_points: []
      world_points: []
```

- `type` は `rect`（左上・右下の2点）または `polygon`（頂点は3点以上）
- 角度は度。画像座標系で**右向きが0度、下向きが90度**（`atan2(dy, dx)` を度に
  したもの。cv2 の y 軸は下向き）。`[290, 60]` のように0をまたぐ区間を書ける
- `exclude_regions` / `transition_zones` / `homography` はいずれも省略できる。
  定義の途中でも読み込みが通るので、書いては確認する作業を繰り返せる
- `image_size` を書いておくと、座標が画像の外に出たときに読み込みで弾かれる

検証は明らかな誤り（点数、画像範囲外、角度の範囲、`id` の重複）に限っている。
違反したときはどのカメラのどの領域かを含めた `CameraConfigError` になる。

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

`local_track_id` はカメラ内でのみ一意。カメラをまたいだ対応と Global ID は
`reid` の出力に記録する。

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
