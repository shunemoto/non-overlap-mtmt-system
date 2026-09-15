"""コマンドラインエントリポイント。"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["build_parser", "main"]

DEFAULT_OUT_ROOT = "outputs"
DEFAULT_CONFIG = "configs/params.yaml"
DEFAULT_CAMERAS = "configs/cameras.yaml"
ALL_CAMERAS = ("camA", "camB", "camC", "camD", "camE", "camF", "camG", "camH")
ALL_DATES = ("0720", "0727", "0730", "0803")
DEFAULT_VIDEO_ROOT = "data/videos"


def _video_path(args: argparse.Namespace, camera: str | None = None) -> Path:
    """--video 未指定なら data/videos/{date}/{camera}.mp4 を使う。"""
    if getattr(args, "video", None):
        return Path(args.video)
    return Path(DEFAULT_VIDEO_ROOT) / args.date / f"{camera or args.camera}.mp4"


def _cameras(args: argparse.Namespace) -> list[str]:
    """--all-cameras / --all-dates が付いていれば全8台、でなければ1台。"""
    if getattr(args, "all_cameras", False) or getattr(args, "all_dates", False):
        return list(ALL_CAMERAS)
    if not getattr(args, "camera", None):
        raise SystemExit("--camera か --all-cameras のどちらかを指定してください")
    return [args.camera]


def cmd_track(args: argparse.Namespace) -> int:
    # ultralytics の import は重いので、サブコマンドが決まってから読む
    from mct.config import tracking_params
    from mct.track import run_track

    params = tracking_params(
        args.config,
        device=args.device,
        stride=args.stride,
        model_pose=args.model_pose,
        model_seg=args.model_seg,
    )
    run_track(
        video=_video_path(args),
        date=args.date,
        camera=args.camera,
        out_root=args.out,
        params=params,
        max_frames=args.max_frames,
    )
    return 0


def _given(**kwargs: object) -> dict[str, object]:
    """未指定（None）のオプションを落とし、呼び出し先の既定値を活かす。"""
    return {k: v for k, v in kwargs.items() if v is not None}


def cmd_viz_video(args: argparse.Namespace) -> int:
    from mct.viz import viz_video

    viz_video(
        _video_path(args), args.date, args.camera, args.out,
        **_given(jump_threshold=args.jump_threshold),
    )
    return 0


def cmd_viz_trajectory(args: argparse.Namespace) -> int:
    from mct.viz import viz_trajectory

    viz_trajectory(_video_path(args), args.date, args.camera, args.out)
    return 0


def cmd_viz_density(args: argparse.Namespace) -> int:
    from mct.viz import viz_density

    viz_density(
        _video_path(args), args.date, args.camera, args.out,
        **_given(bin_px=args.bin_px),
    )
    return 0


def cmd_viz_endpoints(args: argparse.Namespace) -> int:
    from mct.viz import viz_endpoints

    viz_endpoints(_video_path(args), args.date, args.camera, args.out)
    return 0


def cmd_viz_region(args: argparse.Namespace) -> int:
    from mct.viz import viz_region

    for camera in _cameras(args):
        viz_region(
            _video_path(args, camera), args.date, camera, args.out, args.config
        )
    return 0


def cmd_viz_zones(args: argparse.Namespace) -> int:
    from mct.viz import viz_zones

    for camera in _cameras(args):
        viz_zones(_video_path(args, camera), args.date, camera, args.out, args.config)
    return 0


def cmd_zone_report(args: argparse.Namespace) -> int:
    from mct.config import CameraConfig, load_cameras
    from mct.io_utils import load_tracking_result, output_dir
    from mct.pipeline import FILTERED_FILENAME, _filtering_params
    from mct.zone_report import write_zone_report

    cameras = load_cameras(args.config)
    params = _filtering_params(args.params)
    dates = ALL_DATES if args.all_dates else [args.date]

    for date in dates:
        for camera in _cameras(args):
            source = output_dir(args.out, date, camera) / FILTERED_FILENAME
            if not source.is_file():
                print(f"{date}/{camera}: {FILTERED_FILENAME} が無いので飛ばします")
                continue
            write_zone_report(
                date, camera, load_tracking_result(source),
                cameras.get(camera) or CameraConfig(name=camera), params, args.out,
            )
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    from mct.diagnostics import print_summary
    from mct.pipeline import run

    dates = ALL_DATES if args.all_dates else [args.date]
    reports = run(
        dates, _cameras(args), args.out, args.config, args.params, force=args.force
    )
    print_summary(reports)
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    from mct.cache import clear, list_entries

    if args.cache_command == "list":
        entries = list_entries(args.layer)
        if not entries:
            print("キャッシュはありません。")
            return 0
        total = 0
        for layer, path, size in entries:
            print(f"  {layer:<4} {path.name:<40} {size/1024:8.1f} KB")
            total += size
        print(f"  {len(entries)} 件 / 合計 {total/1024/1024:.1f} MB")
        return 0

    n = clear(args.layer)
    print(f"{n} 層のキャッシュを削除しました。" if n else "削除するキャッシュはありません。")
    return 0


def _add_common(
    parser: argparse.ArgumentParser, *, with_config: bool, camera_required: bool = True
) -> None:
    parser.add_argument("--date", required=True, help='撮影日 (例: "0720")')
    # --all-cameras を持つコマンドでは --camera を必須にしない
    parser.add_argument("--camera", required=camera_required, help='カメラID (例: "camA")')
    # --out-root / --params はフェーズ0 で使っていた名前。既存の呼び出しを壊さない。
    parser.add_argument(
        "--out", "--out-root",
        dest="out",
        default=DEFAULT_OUT_ROOT,
        help=f"出力ルートディレクトリ (default: {DEFAULT_OUT_ROOT})",
    )
    parser.add_argument(
        "--video",
        help=f"入力動画のパス (default: {DEFAULT_VIDEO_ROOT}/<date>/<camera>.mp4)",
    )
    if with_config:
        parser.add_argument(
            "--config", "--params",
            dest="config",
            default=DEFAULT_CONFIG,
            help=f"パラメータ YAML (default: {DEFAULT_CONFIG})",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mct.cli",
        description="非重複視野マルチカメラ人物追跡システム",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_track = sub.add_parser("track", help="1カメラ分の映像を追跡し結果を保存する")
    _add_common(p_track, with_config=True)
    p_track.add_argument("--device", help='推論デバイス (例: "mps", "cpu")')
    p_track.add_argument("--stride", type=int, help="N フレームに1枚だけ推論する")
    p_track.add_argument("--model-pose", help="1パス目のモデル")
    p_track.add_argument("--model-seg", help="2パス目のモデル")
    p_track.add_argument(
        "--max-frames", type=int,
        help="先頭 N フレームで打ち切る（スモークテスト用）",
    )
    p_track.set_defaults(func=cmd_track)

    p_viz_v = sub.add_parser(
        "viz-video", help="追跡結果を重畳した動画を書き出す（無人フレームは除外）"
    )
    _add_common(p_viz_v, with_config=False)
    # 既定値は mct.viz 側に持たせている。ここで default を書くと二重定義に
    # なるので、未指定なら None のまま渡さない。
    p_viz_v.add_argument(
        "--jump-threshold", type=float,
        help="足元点がこの px 以上跳んだら JUMP として赤枠にする (default: 150)",
    )
    p_viz_v.set_defaults(func=cmd_viz_video)

    p_viz_t = sub.add_parser("viz-trajectory", help="足元点の軌跡図を書き出す")
    _add_common(p_viz_t, with_config=False)
    p_viz_t.set_defaults(func=cmd_viz_trajectory)

    p_viz_d = sub.add_parser(
        "viz-density", help="足元点の密度ヒートマップを書き出す（観測区間の外縁用）"
    )
    _add_common(p_viz_d, with_config=False)
    p_viz_d.add_argument(
        "--bin-px", type=int,
        help="ヒストグラムのビン幅 [px] (default: 20)",
    )
    p_viz_d.set_defaults(func=cmd_viz_density)

    p_viz_e = sub.add_parser(
        "viz-endpoints", help="各トラックの始点・終点と進行方向を書き出す（遷移領域用）"
    )
    _add_common(p_viz_e, with_config=False)
    p_viz_e.set_defaults(func=cmd_viz_endpoints)

    p_viz_r = sub.add_parser(
        "viz-region", help="定義した観測区間・遷移領域を重ねた図を書き出す"
    )
    _add_common(p_viz_r, with_config=False, camera_required=False)
    p_viz_r.add_argument(
        "--config", default=DEFAULT_CAMERAS,
        help=f"カメラ定義 YAML (default: {DEFAULT_CAMERAS})",
    )
    p_viz_r.add_argument(
        "--all-cameras", action="store_true", help="camA〜camH をまとめて処理する"
    )
    p_viz_r.set_defaults(func=cmd_viz_region)

    p_viz_z = sub.add_parser(
        "viz-zones", help="トラックの出現・退出点を該当ゾーンの色で塗り分けた図を書き出す"
    )
    _add_common(p_viz_z, with_config=False, camera_required=False)
    p_viz_z.add_argument(
        "--config", default=DEFAULT_CAMERAS,
        help=f"カメラ定義 YAML (default: {DEFAULT_CAMERAS})",
    )
    p_viz_z.add_argument(
        "--all-cameras", action="store_true", help="camA〜camH をまとめて処理する"
    )
    p_viz_z.set_defaults(func=cmd_viz_zones)

    p_zr = sub.add_parser(
        "zone-report", help="トラックごとのゾーン該当結果を CSV に書き出す"
    )
    p_zr.add_argument("--date", help='撮影日 (例: "0720")')
    p_zr.add_argument("--camera", help='カメラID (例: "camA")')
    p_zr.add_argument(
        "--all-cameras", action="store_true", help="その日の camA〜camH をまとめて処理する"
    )
    p_zr.add_argument(
        "--all-dates", action="store_true", help="全4日 × camA〜camH を処理する"
    )
    p_zr.add_argument(
        "--out", "--out-root", dest="out", default=DEFAULT_OUT_ROOT,
        help=f"出力ルートディレクトリ (default: {DEFAULT_OUT_ROOT})",
    )
    p_zr.add_argument(
        "--config", default=DEFAULT_CAMERAS,
        help=f"カメラ定義 YAML (default: {DEFAULT_CAMERAS})",
    )
    p_zr.add_argument(
        "--params", default=DEFAULT_CONFIG,
        help=f"パラメータ YAML (default: {DEFAULT_CONFIG})",
    )
    p_zr.set_defaults(func=cmd_zone_report)

    p_filter = sub.add_parser(
        "filter", help="観測区間を適用し静止誤検出を除いて tracks_filtered.json を書く"
    )
    p_filter.add_argument("--date", help='撮影日 (例: "0720")')
    p_filter.add_argument("--camera", help='カメラID (例: "camA")')
    p_filter.add_argument(
        "--all-cameras", action="store_true", help="その日の camA〜camH をまとめて処理する"
    )
    p_filter.add_argument(
        "--all-dates", action="store_true", help="全4日 × camA〜camH を処理する"
    )
    p_filter.add_argument(
        "--out", "--out-root", dest="out", default=DEFAULT_OUT_ROOT,
        help=f"出力ルートディレクトリ (default: {DEFAULT_OUT_ROOT})",
    )
    p_filter.add_argument(
        "--config", default=DEFAULT_CAMERAS,
        help=f"カメラ定義 YAML (default: {DEFAULT_CAMERAS})",
    )
    p_filter.add_argument(
        "--params", default=DEFAULT_CONFIG,
        help=f"パラメータ YAML (default: {DEFAULT_CONFIG})",
    )
    p_filter.add_argument(
        "--force", action="store_true", help="キャッシュを無視して作り直す"
    )
    p_filter.set_defaults(func=cmd_filter)

    p_cache = sub.add_parser("cache", help="中間結果のキャッシュを確認・削除する")
    cache_sub = p_cache.add_subparsers(dest="cache_command", required=True, metavar="ACTION")
    p_cache_list = cache_sub.add_parser("list", help="キャッシュの一覧を表示する")
    p_cache_list.add_argument("--layer", help='層を絞る (例: "L2")')
    p_cache_list.set_defaults(func=cmd_cache)
    p_cache_clear = cache_sub.add_parser("clear", help="キャッシュを削除する")
    p_cache_clear.add_argument("--layer", help='層を絞る (例: "L2")')
    p_cache_clear.set_defaults(func=cmd_cache)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
