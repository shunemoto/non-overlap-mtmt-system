"""コマンドラインエントリポイント。"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["build_parser", "main"]

DEFAULT_OUT_ROOT = "outputs"
DEFAULT_CONFIG = "configs/params.yaml"
DEFAULT_VIDEO_ROOT = "data/videos"


def _video_path(args: argparse.Namespace) -> Path:
    """--video 未指定なら data/videos/{date}/{camera}.mp4 を使う。"""
    if getattr(args, "video", None):
        return Path(args.video)
    return Path(DEFAULT_VIDEO_ROOT) / args.date / f"{args.camera}.mp4"


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


def _add_common(parser: argparse.ArgumentParser, *, with_config: bool) -> None:
    parser.add_argument("--date", required=True, help='撮影日 (例: "0720")')
    parser.add_argument("--camera", required=True, help='カメラID (例: "camA")')
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
