"""コマンドラインエントリポイント。

段階1ではサブコマンドの構造だけを定義し、中身は未実装。
"""

from __future__ import annotations

import argparse

__all__ = ["build_parser", "main"]

DEFAULT_OUT_ROOT = "outputs"
DEFAULT_PARAMS = "configs/params.yaml"


def cmd_track(args: argparse.Namespace) -> int:
    raise NotImplementedError("track is not implemented yet (stage 1: schema only)")


def cmd_viz_video(args: argparse.Namespace) -> int:
    raise NotImplementedError("viz-video is not implemented yet (stage 1: schema only)")


def cmd_viz_trajectory(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "viz-trajectory is not implemented yet (stage 1: schema only)"
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help=f"出力ルートディレクトリ (default: {DEFAULT_OUT_ROOT})",
    )
    parser.add_argument(
        "--params",
        default=DEFAULT_PARAMS,
        help=f"パラメータ YAML (default: {DEFAULT_PARAMS})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mct.cli",
        description="非重複視野マルチカメラ人物追跡システム",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_track = sub.add_parser("track", help="1カメラ分の映像を追跡し結果を保存する")
    p_track.add_argument("--video", required=True, help="入力動画のパス")
    p_track.add_argument("--date", required=True, help='撮影日 (例: "0720")')
    p_track.add_argument("--camera", required=True, help='カメラID (例: "camA")')
    _add_common(p_track)
    p_track.set_defaults(func=cmd_track)

    p_viz_v = sub.add_parser("viz-video", help="追跡結果を重畳した動画を書き出す")
    p_viz_v.add_argument("--date", required=True, help='撮影日 (例: "0720")')
    p_viz_v.add_argument("--camera", required=True, help='カメラID (例: "camA")')
    _add_common(p_viz_v)
    p_viz_v.set_defaults(func=cmd_viz_video)

    p_viz_t = sub.add_parser("viz-trajectory", help="足元点の軌跡図を書き出す")
    p_viz_t.add_argument("--date", required=True, help='撮影日 (例: "0720")')
    p_viz_t.add_argument("--camera", required=True, help='カメラID (例: "camA")')
    _add_common(p_viz_t)
    p_viz_t.set_defaults(func=cmd_viz_trajectory)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
