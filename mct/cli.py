"""コマンドラインエントリポイント。"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["build_parser", "main"]

DEFAULT_OUT_ROOT = "outputs"
GT_OUT_ROOT = "outputs_gt"
DEFAULT_CONFIG = "configs/params.yaml"
DEFAULT_CAMERAS = "configs/cameras.yaml"
ALL_CAMERAS = ("camA", "camB", "camC", "camD", "camE", "camF", "camG", "camH")
ALL_DATES = ("0720", "0727", "0730", "0803")
DEFAULT_VIDEO_ROOT = "data/videos"
DEFAULT_GT_ROOT = "data/gt"
MANUAL_TRANSITIONS = "configs/transitions.yaml"
# 校正済みの実効距離を既定にする（無ければ手動測定値に戻る）
DEFAULT_TRANSITIONS = "configs/transitions_calibrated.yaml"


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


def _resolve_out(args: argparse.Namespace) -> None:
    """--use-gt が付いていれば出力ルートを outputs_gt に切り替える。

    GT 由来のトラックは予測側と同じ TrackingResult 形式なので、読む場所を
    変えるだけで以降の処理はそのまま通る。--out を明示した場合はそちらを優先。
    """
    if getattr(args, "use_gt", False) and args.out == DEFAULT_OUT_ROOT:
        args.out = GT_OUT_ROOT


def _load_gt(args, date):
    """params.yaml のフレームオフセットを当てて GT を読む。"""
    from mct.config import load_params
    from mct.gt import gt_frame_offsets, load_ground_truth

    params = load_params(getattr(args, "params", DEFAULT_CONFIG))
    return load_ground_truth(
        args.gt_root, date, frame_offsets=gt_frame_offsets(params, date)
    )


def _transitions_path(args) -> str:
    """遷移定義のパス。校正済みがあればそちらを使う。

    カメラ間距離は正解遷移から逆算した実効距離の方が到着時刻をよく当てる。
    A->B と B->A で値が違うのは、カメラごとに速度計測の精度が違うことを
    織り込んだ結果なので、そのまま使ってよい。
    """
    chosen = getattr(args, "transitions", None) or DEFAULT_TRANSITIONS
    if chosen == DEFAULT_TRANSITIONS and not Path(chosen).is_file():
        return MANUAL_TRANSITIONS
    return chosen


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
    _resolve_out(args)
    from mct.viz import viz_zones

    for camera in _cameras(args):
        viz_zones(_video_path(args, camera), args.date, camera, args.out, args.config)
    return 0


def cmd_zone_report(args: argparse.Namespace) -> int:
    _resolve_out(args)
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
    _resolve_out(args)
    from mct.diagnostics import print_summary
    from mct.pipeline import run

    dates = ALL_DATES if args.all_dates else [args.date]
    reports = run(
        dates, _cameras(args), args.out, args.config, args.params, force=args.force
    )
    print_summary(reports)
    return 0


def _gt_global_ids_as_predictions(gt) -> dict[str, dict[str, str]]:
    """GT の Global ID をそのまま予測とみなす（動作確認用）。"""
    assignments: dict[str, dict[str, str]] = {}
    for (camera, track_id), global_id in gt.global_ids.items():
        assignments.setdefault(camera, {})[str(track_id)] = global_id
    return assignments


def cmd_eval(args: argparse.Namespace) -> int:
    _resolve_out(args)
    from mct.config import load_params
    from mct.evaluation import (
        evaluate, match_tracks, print_eval_result, print_iou_distribution, print_summary,
    )
    from mct.gt import load_ground_truth
    from mct.io_utils import load_tracking_result
    from mct.predictions import load_predictions

    params = load_params(args.params).get("evaluation") or {}
    dates = ALL_DATES if args.all_dates else [args.date]
    results = []

    for date in dates:
        gt = _load_gt(args, date)

        if args.use_gt_tracks:
            assignments = _gt_global_ids_as_predictions(gt)
            result = evaluate(assignments, gt, use_gt_tracks=True)
        else:
            path = args.pred or (args.pred_pattern or "").format(date=date)
            if not path:
                raise SystemExit("--pred か --pred-pattern、または --use-gt-tracks を指定してください")
            assignments = load_predictions(path).assignments

            matchings = {}
            for camera in ALL_CAMERAS:
                source = Path(args.out) / date / camera / args.pred_tracks
                if not source.is_file():
                    continue
                matchings[camera] = match_tracks(
                    load_tracking_result(source), gt, camera, params
                )
            result = evaluate(assignments, gt, matchings)

        print_eval_result(result)
        results.append(result)

    print_summary(results)
    if args.report_iou:
        print_iou_distribution(results)
    return 0


def cmd_eval_candidates(args: argparse.Namespace) -> int:
    import json

    from mct.evaluation import candidate_recall, print_candidate_recall
    from mct.gt import load_ground_truth

    dates = ALL_DATES if args.all_dates else [args.date]
    for date in dates:
        path = args.candidates or (args.candidates_pattern or "").format(date=date)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        candidates = data.get("links", data) if isinstance(data, dict) else data
        print_candidate_recall(date, candidate_recall(candidates, _load_gt(args, date)))
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    _resolve_out(args)
    import warnings

    from mct.candidate_pipeline import (
        _spatiotemporal, build_candidates, print_arrival_error, print_counts,
        print_speed_stats, write_candidates,
    )

    dates = ALL_DATES if args.all_dates else [args.date]
    all_summaries: list = []
    all_candidates: list = []
    totals = {"tracks": 0, "candidates": 0}

    for date in dates:
        summaries, candidates, meta = build_candidates(
            date, args.out, args.config, _transitions_path(args), args.params,
            no_topology=args.no_topology,
        )
        dest = write_candidates(date, args.out, summaries, candidates, meta)
        print_counts(date, summaries, candidates)
        print(f"  wrote {dest}")

        totals["tracks"] += len(summaries)
        totals["candidates"] += len(candidates)
        all_summaries.extend(summaries)
        all_candidates.append((date, summaries, candidates))

    print(f"\n合計: トラック {totals['tracks']} / 候補 {totals['candidates']}")
    print_speed_stats(all_summaries)

    if args.eval:
        _report_recall(args, all_candidates)
    return 0


def _report_recall(args: argparse.Namespace, per_date: list) -> None:
    """GT と突き合わせて候補の再現率と到着時刻の誤差を出す。"""
    import warnings

    from mct.candidate_pipeline import _spatiotemporal, print_arrival_error
    from mct.config import load_cameras, load_params, load_transitions
    from mct.evaluation import (
        candidate_recall, classify_missed, match_tracks, print_missed_breakdown,
    )
    from mct.gt import gt_transitions, load_ground_truth
    from mct.io_utils import load_tracking_result

    cameras = load_cameras(args.config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        routes = load_transitions(_transitions_path(args), cameras)
    ev_params = load_params(args.params).get("evaluation") or {}
    st_params, _ = _spatiotemporal(args.params)
    fps = float(st_params["fps"])

    total_gt = total_covered = 0
    merged_buckets: dict[str, list] = {}

    for date, summaries, candidates in per_date:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gt = _load_gt(args, date)

        matchings = {}
        for camera in ALL_CAMERAS:
            source = Path(args.out) / date / camera / "tracks_filtered.json"
            if source.is_file():
                matchings[camera] = match_tracks(
                    load_tracking_result(source), gt, camera, ev_params
                )

        report = candidate_recall([c.as_link() for c in candidates], gt, matchings)
        total_gt += report.n_gt
        total_covered += report.n_covered
        print(f"\n--- {date} 候補の再現率 ---")
        print(f"  {report.n_covered} / {report.n_gt} = {report.recall:.4f}")

        by_key = {s.key: s for s in summaries}
        buckets = classify_missed(report.missed, gt, by_key, routes, matchings, fps)
        for reason, items in buckets.items():
            merged_buckets.setdefault(reason, []).extend(items)

        # 到着時刻の誤差は、GT で正解とされた対だけを見る
        to_pred = {
            (camera, g): (camera, int(p))
            for camera, m in matchings.items() for p, g in m.pairs.items()
        }
        correct = set()
        for t in gt_transitions(gt):
            a = to_pred.get((t.from_cam, t.from_track))
            b = to_pred.get((t.to_cam, t.to_track))
            if a and b:
                correct.add((a, b))
        print_arrival_error(candidates, by_key, correct, fps)

    print(f"\n{'='*70}\n全日付の候補再現率: {total_covered} / {total_gt} = "
          f"{total_covered/total_gt if total_gt else 0:.4f}\n{'='*70}")
    print_missed_breakdown(merged_buckets, total_gt)


def _correct_pairs_and_summaries(args, date, summaries):
    """GT で正解とされるトラック対と、(camera, track) 索引を返す。"""
    import warnings

    from mct.config import load_params
    from mct.evaluation import match_tracks
    from mct.gt import gt_transitions, load_ground_truth
    from mct.io_utils import load_tracking_result

    ev_params = load_params(args.params).get("evaluation") or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = _load_gt(args, date)

    to_pred = {}
    if getattr(args, "use_gt", False):
        # GT 由来の入力では track_id が GT のものそのままなので、対応付けは恒等。
        # match_tracks を通しても IoU 1.0 で同じ結果になるが、無駄が多い。
        for camera in gt.boxes:
            for gt_track in gt.evaluable(camera):
                to_pred[(camera, gt_track)] = (camera, gt_track)
    else:
        for camera in ALL_CAMERAS:
            source = Path(args.out) / date / camera / "tracks_filtered.json"
            if not source.is_file():
                continue
            m = match_tracks(load_tracking_result(source), gt, camera, ev_params)
            for pred_track, gt_track in m.pairs.items():
                to_pred[(camera, gt_track)] = (camera, int(pred_track))

    correct = set()
    for t in gt_transitions(gt):
        a = to_pred.get((t.from_cam, t.from_track))
        b = to_pred.get((t.to_cam, t.to_track))
        if a and b:
            correct.add((a, b))
    return correct, {s.key: s for s in summaries}


def cmd_calibrate_distance(args: argparse.Namespace) -> int:
    _resolve_out(args)
    import warnings

    from mct.calibrate import (
        YAML_FILENAME, calibrate, collect_samples, print_report, write_csv, write_yaml,
    )
    from mct.candidate_pipeline import _spatiotemporal, build_candidates
    from mct.config import load_cameras, load_transitions

    cameras = load_cameras(args.config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        routes = load_transitions(_transitions_path(args), cameras)
    st_params, _ = _spatiotemporal(args.params)
    fps = float(st_params["fps"])
    min_samples = int(st_params.get("calib_min_samples", 5))

    dates = [d for d in ALL_DATES if d != args.holdout_date]
    if args.holdout_date:
        print(f"校正から除外する日付: {args.holdout_date}（leave-one-day-out）")

    per_date = []
    for date in dates:
        summaries, candidates, _ = build_candidates(
            date, args.out, args.config, _transitions_path(args), args.params
        )
        correct, by_key = _correct_pairs_and_summaries(args, date, summaries)
        per_date.append(collect_samples(candidates, by_key, correct, fps))
        print(f"  {date}: 正解の対 {len(correct)} からサンプルを収集")

    calibrations = calibrate(per_date, routes, min_samples)
    print_report(calibrations, min_samples)

    csv_path = write_csv(calibrations, args.out)
    yaml_path = write_yaml(calibrations, routes, Path(args.config).parent / YAML_FILENAME)
    print(f"\nwrote {csv_path}")
    print(f"wrote {yaml_path}")
    return 0


def _sigma_samples(args, dates):
    """各日付の正解遷移から、到着時刻の誤差を集める。"""
    from mct.candidate_pipeline import _spatiotemporal
    from mct.spatiotemporal import collect_errors
    from mct.st_pipeline import load_candidates

    params, _ = _spatiotemporal(args.params)
    per_date = []
    for date in dates:
        summaries, candidates, _ = load_candidates(date, args.out)
        correct, _ = _correct_pairs_and_summaries(args, date, summaries)
        per_date.append(collect_errors(candidates, summaries, correct, params))
        print(f"  {date}: 正解の対 {len(correct)} から誤差を収集")
    return per_date, params


def cmd_calibrate_sigma(args: argparse.Namespace) -> int:
    _resolve_out(args)
    from mct.spatiotemporal import estimate_sigma
    from mct.st_pipeline import print_sigma_report, write_sigma_csv

    dates = [d for d in ALL_DATES if d != args.holdout_date]
    if args.holdout_date:
        print(f"校正から除外する日付: {args.holdout_date}（leave-one-day-out）")

    per_date, params = _sigma_samples(args, dates)
    _, estimates, pooled = estimate_sigma(per_date, params)

    print_sigma_report(estimates, pooled, int(params.get("sigma_min_samples", 5)))
    print(f"\nwrote {write_sigma_csv(estimates, pooled, args.out)}")
    return 0


def cmd_st_scores(args: argparse.Namespace) -> int:
    _resolve_out(args)
    from mct.candidate_pipeline import _spatiotemporal
    from mct.spatiotemporal import KIND_IN_RANGE, estimate_sigma, score_candidates
    from mct.st_pipeline import load_candidates, write_scores
    from mct.st_report import (
        Labelled, plot_separation, print_competitors, print_error_by_zone,
        print_kind_breakdown, print_ranks, print_self_loop_ranks, print_separation,
    )

    dates = ALL_DATES if args.all_dates else [args.date]
    # σ は全日付から推定する（--holdout-date で除ける）
    calib_dates = [d for d in ALL_DATES if d != args.holdout_date]
    per_date, params = _sigma_samples(args, calib_dates)
    sigma, _, pooled = estimate_sigma(per_date, params)
    print(f"  共通 σ = {pooled:.0f} フレーム / 個別推定 {len(sigma.per_zone)} ゾーン対\n")

    labelled: list[Labelled] = []
    n_imputed = 0
    for date in dates:
        summaries, candidates, _ = load_candidates(date, args.out)
        scores = score_candidates(candidates, summaries, sigma, params)
        dest = write_scores(date, args.out, scores, sigma, params)

        correct, _ = _correct_pairs_and_summaries(args, date, summaries)
        by_kind = {k: 0 for k in ("in_range", "out_of_range", "self_loop")}
        for candidate, score in zip(candidates, scores):
            by_kind[score.kind] += 1
            n_imputed += int(score.speed_imputed)
            zone = None
            if candidate.route is not None:
                r = candidate.route
                zone = (r.from_cam, r.from_zone, r.to_cam, r.to_zone)
            labelled.append(
                Labelled(score, (score.tgt, score.src) in correct, date, zone)
            )
        print(f"{date}: 候補 {len(scores)} 件 -> {dest}")
        print(f"  {by_kind}")

    print(f"\n速度を中央値で代用した候補: {n_imputed} 件")
    print_separation(labelled)
    print_kind_breakdown(labelled)
    print_error_by_zone(labelled, float(params.get("fps", 30)))
    print_competitors(labelled)
    print_ranks(labelled)
    print_self_loop_ranks(labelled)

    png = plot_separation(labelled, Path(args.out) / "viz" / "st_likelihood_separation.png")
    print(f"\nwrote {png}")
    return 0


def cmd_gt_features(args: argparse.Namespace) -> int:
    _resolve_out(args)
    from mct.config import CameraConfig, load_cameras
    from mct.gt_features import extract_features, print_stats
    from mct.io_utils import save_tracking_result
    from mct.pipeline import FILTERED_FILENAME, _filtering_params

    cameras = load_cameras(args.config)
    params = dict(_filtering_params(args.params))
    params.update(_gt_feature_params(args.params))

    dates = ALL_DATES if args.all_dates else [args.date]
    total = 0.0
    for date in dates:
        gt = _load_gt(args, date)
        for camera in _cameras(args):
            video = Path(DEFAULT_VIDEO_ROOT) / date / f"{camera}.mp4"
            result, stats = extract_features(
                video, gt, camera,
                cameras.get(camera) or CameraConfig(name=camera),
                params, args.out,
            )
            print_stats(stats)
            if result is None:
                print("  （GT が空、または残るトラックが無いので出力なし）")
                continue
            dest = Path(args.out) / date / camera / FILTERED_FILENAME
            save_tracking_result(result, dest)
            print(f"  wrote {dest}")
            total += stats.seconds

    print(f"\n合計 {total/60:.1f} 分")
    return 0


def _gt_feature_params(params_path) -> dict:
    """GT からの特徴抽出で使う、追跡側と共通の設定。"""
    from mct.config import tracking_params

    tracking = tracking_params(params_path)
    return {
        key: tracking[key]
        for key in (
            "model_pose", "model_seg", "crop_padding_ratio", "imgsz",
            "device", "jpeg_quality", "mask_iou_threshold",
        )
    }


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
    p_viz_z.add_argument(
        "--use-gt", action="store_true",
        help=f"GT 由来の特徴（{GT_OUT_ROOT}/）を入力にする",
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
    p_zr.add_argument(
        "--use-gt", action="store_true",
        help=f"GT 由来の特徴（{GT_OUT_ROOT}/）を入力にする",
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
    p_filter.add_argument(
        "--use-gt", action="store_true",
        help=f"GT 由来の特徴（{GT_OUT_ROOT}/）を入力にする",
    )
    p_filter.set_defaults(func=cmd_filter)

    p_cand = sub.add_parser(
        "candidates", help="カメラ間トポロジーから対応候補を列挙する"
    )
    p_cand.add_argument("--date", help='撮影日 (例: "0720")')
    p_cand.add_argument("--all-dates", action="store_true", help="全4日を処理する")
    p_cand.add_argument(
        "--no-topology", action="store_true",
        help="トポロジーを使わず時間順序と区間非重複だけで列挙する（比較手法①）",
    )
    p_cand.add_argument(
        "--eval", action="store_true", help="GT と突き合わせて候補の再現率も出す"
    )
    p_cand.add_argument(
        "--out", "--out-root", dest="out", default=DEFAULT_OUT_ROOT,
        help=f"出力ルートディレクトリ (default: {DEFAULT_OUT_ROOT})",
    )
    p_cand.add_argument(
        "--config", default=DEFAULT_CAMERAS,
        help=f"カメラ定義 YAML (default: {DEFAULT_CAMERAS})",
    )
    p_cand.add_argument(
        "--transitions", default=DEFAULT_TRANSITIONS,
        help=f"遷移定義 YAML (default: {DEFAULT_TRANSITIONS})",
    )
    p_cand.add_argument(
        "--params", default=DEFAULT_CONFIG,
        help=f"パラメータ YAML (default: {DEFAULT_CONFIG})",
    )
    p_cand.add_argument(
        "--gt-root", default=DEFAULT_GT_ROOT, help=f"GT のルート (default: {DEFAULT_GT_ROOT})"
    )
    p_cand.add_argument(
        "--use-gt", action="store_true",
        help=f"GT 由来の特徴（{GT_OUT_ROOT}/）を入力にする",
    )
    p_cand.set_defaults(func=cmd_candidates)

    p_calib = sub.add_parser(
        "calibrate-distance", help="正解遷移からゾーン対ごとの距離を校正する"
    )
    p_calib.add_argument(
        "--holdout-date",
        help="この日付を校正から除く（leave-one-day-out で評価するため）",
    )
    p_calib.add_argument(
        "--out", "--out-root", dest="out", default=DEFAULT_OUT_ROOT,
        help=f"出力ルートディレクトリ (default: {DEFAULT_OUT_ROOT})",
    )
    p_calib.add_argument("--config", default=DEFAULT_CAMERAS)
    p_calib.add_argument("--transitions", default=DEFAULT_TRANSITIONS)
    p_calib.add_argument("--params", default=DEFAULT_CONFIG)
    p_calib.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    p_calib.add_argument(
        "--use-gt", action="store_true",
        help=f"GT 由来の特徴（{GT_OUT_ROOT}/）を入力にする",
    )
    p_calib.set_defaults(func=cmd_calibrate_distance)

    def _add_st_common(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--out", "--out-root", dest="out", default=DEFAULT_OUT_ROOT,
            help=f"出力ルートディレクトリ (default: {DEFAULT_OUT_ROOT})",
        )
        parser.add_argument("--config", default=DEFAULT_CAMERAS)
        parser.add_argument("--transitions", default=DEFAULT_TRANSITIONS)
        parser.add_argument("--params", default=DEFAULT_CONFIG)
        parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
        parser.add_argument(
            "--holdout-date", help="σ の推定からこの日付を除く（leave-one-day-out）"
        )

    p_sigma = sub.add_parser(
        "calibrate-sigma", help="正解遷移から到着時刻のばらつき σ を推定する"
    )
    _add_st_common(p_sigma)
    p_sigma.add_argument("--all-dates", action="store_true", help="（既定で全4日を使う）")
    p_sigma.add_argument(
        "--use-gt", action="store_true",
        help=f"GT 由来の特徴（{GT_OUT_ROOT}/）を入力にする",
    )
    p_sigma.set_defaults(func=cmd_calibrate_sigma)

    p_st = sub.add_parser("st-scores", help="候補に時空間尤度を付けて書き出す")
    p_st.add_argument("--date", help='撮影日 (例: "0720")')
    p_st.add_argument("--all-dates", action="store_true", help="全4日を処理する")
    _add_st_common(p_st)
    p_st.add_argument(
        "--use-gt", action="store_true",
        help=f"GT 由来の特徴（{GT_OUT_ROOT}/）を入力にする",
    )
    p_st.set_defaults(func=cmd_st_scores)

    p_gtf = sub.add_parser(
        "gt-features", help="GT の bbox から crop・マスク・骨格点を作る（単体評価用）"
    )
    p_gtf.add_argument("--date", help='撮影日 (例: "0720")')
    p_gtf.add_argument("--camera", help='カメラID (例: "camA")')
    p_gtf.add_argument("--all-cameras", action="store_true", help="camA〜camH をまとめて処理する")
    p_gtf.add_argument("--all-dates", action="store_true", help="全4日 × camA〜camH を処理する")
    p_gtf.add_argument(
        "--out", "--out-root", dest="out", default=GT_OUT_ROOT,
        help=f"出力ルートディレクトリ (default: {GT_OUT_ROOT})",
    )
    p_gtf.add_argument("--config", default=DEFAULT_CAMERAS)
    p_gtf.add_argument("--params", default=DEFAULT_CONFIG)
    p_gtf.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    p_gtf.set_defaults(func=cmd_gt_features)

    p_eval = sub.add_parser("eval", help="Ground Truth と突き合わせて ID 指標を出す")
    p_eval.add_argument("--date", help='撮影日 (例: "0720")')
    p_eval.add_argument("--all-dates", action="store_true", help="全4日を評価する")
    p_eval.add_argument("--pred", help="予測結果 JSON のパス")
    p_eval.add_argument(
        "--pred-pattern", help='日付を埋めるパターン (例: "outputs/{date}/global_ids.json")'
    )
    p_eval.add_argument(
        "--use-gt-tracks", action="store_true",
        help="GT の Global ID をそのまま予測として使う（再同定単体評価モード）",
    )
    p_eval.add_argument(
        "--pred-tracks", default="tracks_filtered.json",
        help="対応付けに使う追跡結果 (default: tracks_filtered.json)",
    )
    p_eval.add_argument(
        "--report-iou", action="store_true", help="対応付けの IoU 分布を出す"
    )
    p_eval.add_argument(
        "--gt-root", default=DEFAULT_GT_ROOT, help=f"GT のルート (default: {DEFAULT_GT_ROOT})"
    )
    p_eval.add_argument(
        "--out", "--out-root", dest="out", default=DEFAULT_OUT_ROOT,
        help=f"出力ルートディレクトリ (default: {DEFAULT_OUT_ROOT})",
    )
    p_eval.add_argument(
        "--params", default=DEFAULT_CONFIG,
        help=f"パラメータ YAML (default: {DEFAULT_CONFIG})",
    )
    p_eval.add_argument(
        "--use-gt", action="store_true",
        help=f"GT 由来の特徴（{GT_OUT_ROOT}/）を入力にする",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_evc = sub.add_parser(
        "eval-candidates", help="GT の正解遷移が候補に含まれた割合を出す"
    )
    p_evc.add_argument("--date", help='撮影日 (例: "0720")')
    p_evc.add_argument("--all-dates", action="store_true", help="全4日を評価する")
    p_evc.add_argument("--candidates", help="候補 JSON のパス")
    p_evc.add_argument(
        "--candidates-pattern", help='日付を埋めるパターン (例: "outputs/{date}/candidates.json")'
    )
    p_evc.add_argument(
        "--gt-root", default=DEFAULT_GT_ROOT, help=f"GT のルート (default: {DEFAULT_GT_ROOT})"
    )
    p_evc.set_defaults(func=cmd_eval_candidates)

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
