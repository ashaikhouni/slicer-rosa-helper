#!/usr/bin/env python3
"""Offline evaluation for a learned candidate-line ranker.

This tool keeps the geometric proposal stage fixed and only learns how to rank
candidate shank lines. It is intended as a diagnostic: if ranking improves held-
out performance, the current detector is proposal-rich but scored poorly; if not,
proposal coverage is still the main bottleneck.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval_seeg_localization import (  # type: ignore
    PredictedShank,
    build_detection_context,
    compare_shanks,
    default_detection_config,
    iter_subject_rows,
    load_ground_truth_shanks,
    match_shanks,
)


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n <= 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return arr / n


def _line_distance(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> float:
    u = _unit(d0)
    v = _unit(d1)
    w0 = p0 - p1
    c = np.cross(u, v)
    cn = float(np.linalg.norm(c))
    if cn <= 1e-6:
        return float(np.linalg.norm(np.cross(w0, u)))
    return float(abs(np.dot(w0, c)) / cn)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(default) if not math.isfinite(out) else out


def candidate_to_predicted_shank(candidate: dict[str, Any], name: str | None = None) -> PredictedShank:
    start = tuple(float(v) for v in list(candidate.get("start_ras") or [0.0, 0.0, 0.0])[:3])
    end = tuple(float(v) for v in list(candidate.get("end_ras") or [0.0, 0.0, 0.0])[:3])
    return PredictedShank(
        name=str(name or candidate.get("name") or "candidate"),
        start_ras=start,
        end_ras=end,
        direction_ras=tuple(float(v) for v in _unit(np.asarray(end, dtype=float) - np.asarray(start, dtype=float))),
        length_mm=_safe_float(candidate.get("length_mm"), 0.0),
        support_count=int(candidate.get("inlier_count", 0)),
        confidence=_safe_float(candidate.get("selection_score"), 0.0),
    )


def extract_candidate_lines(result: dict[str, Any]) -> list[dict[str, Any]]:
    extras = dict((result.get("meta") or {}).get("extras") or {})
    candidates = list(extras.get("candidate_lines") or [])
    out: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates, start=1):
        row = dict(cand)
        row.setdefault("name", f"C{idx:03d}")
        out.append(row)
    return out


def candidate_feature_vector(candidate: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    role = str(candidate.get("seed_role") or "unknown").strip().lower()
    length_mm = _safe_float(candidate.get("length_mm"), 0.0)
    support_weight = _safe_float(candidate.get("support_weight"), 0.0)
    inlier_count = _safe_float(candidate.get("inlier_count"), 0.0)
    inside_fraction = _safe_float(candidate.get("inside_fraction"), 0.0)
    depth_span_mm = _safe_float(candidate.get("depth_span_mm"), 0.0)
    rms_mm = _safe_float(candidate.get("rms_mm"), 0.0)
    selection_score = _safe_float(candidate.get("selection_score"), 0.0)
    entry_depth_mm = _safe_float(candidate.get("entry_depth_mm"), 0.0)
    target_depth_mm = _safe_float(candidate.get("target_depth_mm"), 0.0)

    feats = {
        "selection_score": selection_score,
        "log_support_weight": math.log1p(max(0.0, support_weight)),
        "log_inlier_count": math.log1p(max(0.0, inlier_count)),
        "inside_fraction": inside_fraction,
        "depth_span_mm": depth_span_mm,
        "length_mm": length_mm,
        "rms_mm": rms_mm,
        "entry_depth_mm": entry_depth_mm,
        "target_depth_mm": target_depth_mm,
        "role_core": 1.0 if role == "core" else 0.0,
        "role_rescue": 1.0 if role == "rescue" else 0.0,
        "role_unknown": 1.0 if role not in {"core", "rescue"} else 0.0,
    }
    return np.asarray(list(feats.values()), dtype=float), feats


FEATURE_NAMES = list(candidate_feature_vector({})[1].keys())


def label_candidate(candidate: dict[str, Any], gt_shanks: list[Any], *, label_end_mm: float, label_start_mm: float, label_angle_deg: float) -> tuple[int, dict[str, float]]:
    pred = candidate_to_predicted_shank(candidate)
    if not gt_shanks:
        return 0, {"best_end_error_mm": float("inf"), "best_start_error_mm": float("inf"), "best_angle_deg": float("inf")}
    best = None
    for gt in gt_shanks:
        pm = compare_shanks(gt, pred)
        if best is None or pm.score < best.score:
            best = pm
    assert best is not None
    label = int(best.end_error_mm <= label_end_mm and best.start_error_mm <= label_start_mm and best.angle_deg <= label_angle_deg)
    return label, {
        "best_end_error_mm": float(best.end_error_mm),
        "best_start_error_mm": float(best.start_error_mm),
        "best_angle_deg": float(best.angle_deg),
    }


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, *, steps: int = 1200, lr: float = 0.05, l2: float = 1e-3) -> tuple[np.ndarray, float, dict[str, float]]:
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("empty training design matrix")
    y = y.astype(float).reshape(-1)
    pos = float(np.sum(y >= 0.5))
    neg = float(y.shape[0] - pos)
    pos_weight = 1.0 if pos <= 0.0 or neg <= 0.0 else neg / pos
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    for _ in range(int(steps)):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        sample_weight = np.where(y >= 0.5, pos_weight, 1.0)
        err = (p - y) * sample_weight
        grad_w = (X.T @ err) / float(X.shape[0]) + float(l2) * w
        grad_b = float(np.mean(err))
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
    return w, float(b), {"positive_count": pos, "negative_count": neg, "positive_weight": pos_weight}


def score_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = X @ w + float(b)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu, sigma


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu.reshape(1, -1)) / sigma.reshape(1, -1)


def select_top_non_overlapping(candidates: list[dict[str, Any]], score_key: str, *, target_count: int, angle_deg: float, line_distance_mm: float) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda row: float(row.get(score_key, 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in ordered:
        s0 = np.asarray(cand.get("start_ras") or [0.0, 0.0, 0.0], dtype=float)
        e0 = np.asarray(cand.get("end_ras") or [0.0, 0.0, 0.0], dtype=float)
        d0 = _unit(e0 - s0)
        is_dup = False
        for prev in kept:
            s1 = np.asarray(prev.get("start_ras") or [0.0, 0.0, 0.0], dtype=float)
            e1 = np.asarray(prev.get("end_ras") or [0.0, 0.0, 0.0], dtype=float)
            d1 = _unit(e1 - s1)
            ang = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(d0, d1))), 0.0, 1.0))))
            dist = _line_distance(0.5 * (s0 + e0), d0, 0.5 * (s1 + e1), d1)
            if ang <= angle_deg and dist <= line_distance_mm:
                is_dup = True
                break
        if is_dup:
            continue
        kept.append(cand)
        if len(kept) >= int(target_count):
            break
    return kept


def evaluate_selected(subject_id: str, gt_shanks: list[Any], selected: list[dict[str, Any]], *, match_distance_mm: float, match_start_mm: float, match_angle_deg: float) -> dict[str, Any]:
    preds = [candidate_to_predicted_shank(c, name=f"{subject_id}_{idx:02d}") for idx, c in enumerate(selected, start=1)]
    assignments, summary = match_shanks(
        gt_shanks,
        preds,
        match_distance_mm=match_distance_mm,
        match_start_mm=match_start_mm,
        match_angle_deg=match_angle_deg,
    )
    return {
        "pred_count": len(preds),
        "matched": int(summary["matched"]),
        "false_negative": int(summary["false_negative"]),
        "false_positive": int(summary["false_positive"]),
        "recall": 0.0 if not gt_shanks else float(summary["matched"]) / float(len(gt_shanks)),
        "precision": 0.0 if not preds else float(summary["matched"]) / float(len(preds)),
        "assignments": assignments,
    }


def collect_subject_candidates(row: dict[str, str], *, pipeline_key: str = "contact_pitch_v1", config: dict[str, Any], extras: dict[str, Any], label_end_mm: float, label_start_mm: float, label_angle_deg: float) -> dict[str, Any]:
    subject_id = str(row["subject_id"])
    gt_shanks = load_ground_truth_shanks(row["labels_path"], row.get("shanks_path"))
    run_cfg = dict(config)
    run_cfg["return_candidate_lines"] = True
    ctx, _ = build_detection_context(row["ct_path"], run_id=f"ranker_{subject_id}_{pipeline_key}", config=run_cfg, extras=dict(extras))
    result = run_contact_pitch_v1(ctx)
    if str(result.get("status", "ok")).lower() == "error":
        err = dict(result.get("error") or {})
        raise RuntimeError(f"{subject_id}: {err.get('message', 'Detection failed')} (stage={err.get('stage', 'pipeline')})")
    candidates = extract_candidate_lines(result)
    feature_rows: list[dict[str, Any]] = []
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for idx, cand in enumerate(candidates, start=1):
        x, feat_map = candidate_feature_vector(cand)
        label, label_info = label_candidate(
            cand,
            gt_shanks,
            label_end_mm=label_end_mm,
            label_start_mm=label_start_mm,
            label_angle_deg=label_angle_deg,
        )
        X_rows.append(x)
        y_rows.append(int(label))
        feature_rows.append(
            {
                "candidate_id": f"{subject_id}_C{idx:03d}",
                "subject_id": subject_id,
                "label": int(label),
                **feat_map,
                **label_info,
            }
        )
    X = np.asarray(X_rows, dtype=float) if X_rows else np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    y = np.asarray(y_rows, dtype=int) if y_rows else np.zeros((0,), dtype=int)
    return {
        "subject_id": subject_id,
        "gt_shanks": gt_shanks,
        "result": result,
        "candidates": candidates,
        "features": X,
        "labels": y,
        "feature_rows": feature_rows,
    }


def _loocv_splits(subject_ids: list[str]) -> list[tuple[list[str], list[str]]]:
    return [([sid for sid in subject_ids if sid != holdout], [holdout]) for holdout in subject_ids]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a learned candidate-line ranker on held-out subjects")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pipeline-key", default="blob_persistence_v2")
    p.add_argument("--subjects", default="T1,T2,T3,T25", help="Comma-separated subject ids")
    p.add_argument("--selection-mode", choices=["gt_count", "fixed"], default="gt_count")
    p.add_argument("--selection-count", type=int, default=12)
    p.add_argument("--selection-nms-angle-deg", type=float, default=8.0)
    p.add_argument("--selection-nms-line-distance-mm", type=float, default=2.5)
    p.add_argument("--label-end-mm", type=float, default=12.0)
    p.add_argument("--label-start-mm", type=float, default=20.0)
    p.add_argument("--label-angle-deg", type=float, default=10.0)
    p.add_argument("--match-distance-mm", type=float, default=4.0)
    p.add_argument("--match-start-mm", type=float, default=15.0)
    p.add_argument("--match-angle-deg", type=float, default=25.0)
    p.add_argument("--proposal-seed-limit", type=int, default=180)
    p.add_argument("--metal-threshold-hu", type=float, default=1800.0)
    p.add_argument("--use-head-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--build-head-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--head-mask-threshold-hu", type=float, default=-500.0)
    p.add_argument("--head-mask-close-mm", type=float, default=2.0)
    p.add_argument("--head-mask-method", choices=["outside_air", "not_air_lcc"], default="outside_air")
    p.add_argument("--head-mask-aggressive-cleanup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--head-gate-erode-vox", type=int, default=1)
    p.add_argument("--head-gate-dilate-vox", type=int, default=1)
    p.add_argument("--head-gate-margin-mm", type=float, default=0.0)
    p.add_argument("--min-metal-depth-mm", type=float, default=5.0)
    p.add_argument("--max-metal-depth-mm", type=float, default=220.0)
    p.add_argument("--max-points", type=int, default=300000)
    p.add_argument("--max-lines", type=int, default=40)
    p.add_argument("--inlier-radius-mm", type=float, default=1.2)
    p.add_argument("--min-length-mm", type=float, default=20.0)
    p.add_argument("--min-inliers", type=int, default=6)
    p.add_argument("--ransac-iterations", type=int, default=240)
    p.add_argument("--min-blob-voxels", type=int, default=2)
    p.add_argument("--max-blob-voxels", type=int, default=1200)
    p.add_argument("--min-blob-peak-hu", type=float, default=None)
    p.add_argument("--use-distance-mask-for-blob-candidates", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-rescue-pass", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rescue-min-inliers-scale", type=float, default=0.6)
    p.add_argument("--rescue-max-lines", type=int, default=6)
    p.add_argument("--use-model-score", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-model-score", type=float, default=0.10)
    p.add_argument("--electrode-library", default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    subject_filter = {s.strip() for s in str(args.subjects).split(",") if s.strip()}
    rows = iter_subject_rows(dataset_root, subject_filter)
    if not rows:
        raise SystemExit("No subjects matched the requested filter")

    if not hasattr(args, "selection_target_count"):
        args.selection_target_count = None
    config = default_detection_config(args)
    config["proposal_seed_limit"] = int(args.proposal_seed_limit)
    extras = {"electrode_library": None}

    subject_data: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row["subject_id"])
        subject_data[sid] = collect_subject_candidates(
            row,
            pipeline_key=str(args.pipeline_key),
            config=config,
            extras=extras,
            label_end_mm=float(args.label_end_mm),
            label_start_mm=float(args.label_start_mm),
            label_angle_deg=float(args.label_angle_deg),
        )

    feature_rows = [r for sid in sorted(subject_data) for r in subject_data[sid]["feature_rows"]]
    splits = _loocv_splits(sorted(subject_data))
    fold_rows: list[dict[str, Any]] = []
    assignment_artifacts: dict[str, Any] = {}

    for train_subjects, test_subjects in splits:
        X_train = np.concatenate([subject_data[sid]["features"] for sid in train_subjects], axis=0)
        y_train = np.concatenate([subject_data[sid]["labels"] for sid in train_subjects], axis=0)
        mu, sigma = standardize_fit(X_train)
        X_train_std = standardize_apply(X_train, mu, sigma)
        w, b, train_stats = fit_logistic_regression(X_train_std, y_train)

        for sid in test_subjects:
            data = subject_data[sid]
            X_test = data["features"]
            X_test_std = standardize_apply(X_test, mu, sigma) if X_test.size else X_test
            learned_scores = score_logistic(X_test_std, w, b) if X_test.size else np.zeros((0,), dtype=float)
            candidates = [dict(c) for c in data["candidates"]]
            for idx, cand in enumerate(candidates):
                cand["baseline_rank_score"] = _safe_float(cand.get("selection_score"), 0.0)
                cand["learned_rank_score"] = float(learned_scores[idx]) if idx < learned_scores.shape[0] else 0.0

            target_count = len(data["gt_shanks"]) if args.selection_mode == "gt_count" else int(args.selection_count)
            baseline_selected = select_top_non_overlapping(
                candidates,
                "baseline_rank_score",
                target_count=target_count,
                angle_deg=float(args.selection_nms_angle_deg),
                line_distance_mm=float(args.selection_nms_line_distance_mm),
            )
            learned_selected = select_top_non_overlapping(
                candidates,
                "learned_rank_score",
                target_count=target_count,
                angle_deg=float(args.selection_nms_angle_deg),
                line_distance_mm=float(args.selection_nms_line_distance_mm),
            )
            baseline_eval = evaluate_selected(
                sid,
                data["gt_shanks"],
                baseline_selected,
                match_distance_mm=float(args.match_distance_mm),
                match_start_mm=float(args.match_start_mm),
                match_angle_deg=float(args.match_angle_deg),
            )
            learned_eval = evaluate_selected(
                sid,
                data["gt_shanks"],
                learned_selected,
                match_distance_mm=float(args.match_distance_mm),
                match_start_mm=float(args.match_start_mm),
                match_angle_deg=float(args.match_angle_deg),
            )
            positives_available = int(np.sum(data["labels"]))
            fold_rows.append(
                {
                    "test_subject": sid,
                    "train_subjects": ",".join(train_subjects),
                    "candidate_count": len(candidates),
                    "label_positive_count": positives_available,
                    "baseline_pred_count": baseline_eval["pred_count"],
                    "baseline_matched": baseline_eval["matched"],
                    "baseline_recall": f"{baseline_eval['recall']:.4f}",
                    "baseline_precision": f"{baseline_eval['precision']:.4f}",
                    "learned_pred_count": learned_eval["pred_count"],
                    "learned_matched": learned_eval["matched"],
                    "learned_recall": f"{learned_eval['recall']:.4f}",
                    "learned_precision": f"{learned_eval['precision']:.4f}",
                    "train_positive_count": int(train_stats["positive_count"]),
                    "train_negative_count": int(train_stats["negative_count"]),
                }
            )
            assignment_artifacts[sid] = {
                "train_subjects": train_subjects,
                "candidate_count": len(candidates),
                "label_positive_count": positives_available,
                "model_weights": {name: float(val) for name, val in zip(FEATURE_NAMES, w.tolist())},
                "model_bias": float(b),
                "baseline_selected": baseline_selected,
                "baseline_eval": baseline_eval,
                "learned_selected": learned_selected,
                "learned_eval": learned_eval,
            }

    with open(out_dir / "feature_rows.json", "w", encoding="utf-8") as f:
        json.dump(feature_rows, f, indent=2)
    with open(out_dir / "fold_results.json", "w", encoding="utf-8") as f:
        json.dump(fold_rows, f, indent=2)
    with open(out_dir / "assignments.json", "w", encoding="utf-8") as f:
        json.dump(assignment_artifacts, f, indent=2)

    aggregate = {
        "subjects": sorted(subject_data),
        "pipeline_key": str(args.pipeline_key),
        "feature_names": FEATURE_NAMES,
        "label_thresholds": {
            "end_mm": float(args.label_end_mm),
            "start_mm": float(args.label_start_mm),
            "angle_deg": float(args.label_angle_deg),
        },
        "match_thresholds": {
            "end_mm": float(args.match_distance_mm),
            "start_mm": float(args.match_start_mm),
            "angle_deg": float(args.match_angle_deg),
        },
        "baseline_matched_total": int(sum(int(r["baseline_matched"]) for r in fold_rows)),
        "learned_matched_total": int(sum(int(r["learned_matched"]) for r in fold_rows)),
        "baseline_pred_total": int(sum(int(r["baseline_pred_count"]) for r in fold_rows)),
        "learned_pred_total": int(sum(int(r["learned_pred_count"]) for r in fold_rows)),
        "gt_total": int(sum(len(subject_data[sid]["gt_shanks"]) for sid in subject_data)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
