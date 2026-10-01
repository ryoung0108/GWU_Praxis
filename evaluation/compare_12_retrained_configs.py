#!/usr/bin/env python3
r"""Compare retrained YOLOv8 n/s/m models on four aligned capture resolutions.

Run on the Windows Praxis machine: py compare_12_retrained_configs.py
Requires opencv-python, numpy, ultralytics. Reads C:\PraxisData\processed.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

RESOLUTIONS = {
    "640x480": (640, 480),
    "1332x990": (1332, 990),
    "2028x1520": (2028, 1520),
    "4056x3040": (4056, 3040),
}
WEIGHTS = {
    size: f"Baseline_Reproduction_20260930/yolov8{size}_03_full_board_best.pt"
    for size in ("n", "s", "m")
}
TRUTH = np.array([
    [3, 0, 7, 1, 4], [2, 6, 5, 8, 9], [4, 9, 7, 3, 9],
    [6, 5, 0, 1, 4], [0, 3, 2, 7, 8], [8, 2, 1, 6, 5],
    [2, 4, 0, 5, 9], [3, 6, 8, 7, 1],
])
REF_CORNERS = np.array([
    [391.6073, 30.5382], [1147.8764, 37.7236],
    [1126.3200, 1528.7055], [395.2000, 1532.2982],
], dtype=np.float32)
GRID = np.array([[0, 0], [500, 0], [500, 800], [0, 800]], dtype=np.float32)
SOURCE_W, SOURCE_H = 4056, 3040
DOWNSAMPLE = 4
PATCH_X, PATCH_Y, PATCH_W, PATCH_H = 418, 5, 196, 387


def edge_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (gray.shape[1] // DOWNSAMPLE,
                              gray.shape[0] // DOWNSAMPLE), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def feature_data(frame, mask=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (SOURCE_W//4, SOURCE_H//4))
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    return cv2.SIFT_create(nfeatures=2500).detectAndCompute(gray, mask)


def align(source, template, ref_features, min_match):
    match = cv2.matchTemplate(edge_image(source), template, cv2.TM_CCOEFF_NORMED)
    _, quality, _, (x, y) = cv2.minMaxLoc(match)
    if quality >= min_match:
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, 2] = [(x-PATCH_X)*4, (y-PATCH_Y)*4]
        return matrix, quality, "template", 0
    ref_kp, ref_desc = ref_features
    kp, desc = feature_data(source)
    if desc is None or ref_desc is None:
        return None, quality, "no features", 0
    pairs = cv2.BFMatcher().knnMatch(ref_desc, desc, k=2)
    good = [pair[0] for pair in pairs if len(pair)==2
            and pair[0].distance < .7*pair[1].distance]
    if len(good)<12:
        return None, quality, "too few feature matches", len(good)
    origin = np.float32([ref_kp[m.queryIdx].pt for m in good])
    target = np.float32([kp[m.trainIdx].pt for m in good])
    affine, mask = cv2.estimateAffinePartial2D(
        origin, target, method=cv2.RANSAC, ransacReprojThreshold=2.0,
        maxIters=3000, confidence=.995)
    if affine is None:
        return None, quality, "feature fit failed", 0
    keep = mask.ravel().astype(bool)
    count = int(keep.sum())
    # Require matches spread over the board, not just one digit.
    span = np.ptp(origin[keep], axis=0) if count else np.zeros(2)
    scale = float(np.hypot(affine[0,0], affine[1,0]))
    rotation = float(np.degrees(np.arctan2(affine[1,0], affine[0,0])))
    if (count<12 or count/len(good)<.5 or span[0]<80 or span[1]<180
            or not .8<scale<1.2 or abs(rotation)>15):
        return None, quality, "feature geometry rejected", count
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2] = affine
    matrix[:2,2] *= 4
    return matrix, quality, "features", count


def grid_preview(roi, corners, path):
    preview = roi.copy()
    inverse = cv2.getPerspectiveTransform(GRID, corners.astype(np.float32))
    for vertical in range(6):
        pts = np.float32([[[vertical*100,0],[vertical*100,800]]])
        line = cv2.perspectiveTransform(pts, inverse)[0]
        cv2.line(preview, tuple(np.rint(line[0]).astype(int)),
                 tuple(np.rint(line[1]).astype(int)), (0,255,255), 2)
    for horizontal in range(9):
        pts = np.float32([[[0,horizontal*100],[500,horizontal*100]]])
        line = cv2.perspectiveTransform(pts, inverse)[0]
        cv2.line(preview, tuple(np.rint(line[0]).astype(int)),
                 tuple(np.rint(line[1]).astype(int)), (0,255,255), 2)
    cv2.imwrite(str(path), preview)


def write_rows(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(r"C:\PraxisData"))
    parser.add_argument("--days", nargs="+", required=True,
                        help="Dates YYYY-MM-DD after the board was rearranged")
    parser.add_argument("--min-match", type=float, default=0.75)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--alignment-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    days = set(args.days)
    base = args.base
    out = args.output or base / "results" / "compare_12_retrained_aligned"
    out.mkdir(parents=True, exist_ok=True)

    reference_path = (base / "processed" / "2026-09-24_10-30-00"
                      / "2026-09-24_10-30-08_4056x3040.jpg")
    reference = cv2.imread(str(reference_path))
    if reference is None or reference.shape[:2] != (SOURCE_H, SOURCE_W):
        raise FileNotFoundError(f"Missing or invalid reference: {reference_path}")
    edges = edge_image(reference)
    template = edges[PATCH_Y:PATCH_Y + PATCH_H, PATCH_X:PATCH_X + PATCH_W]
    if template.shape != (PATCH_H, PATCH_W):
        raise ValueError("Reference alignment patch has wrong dimensions")
    board_mask = np.zeros((SOURCE_H//4, SOURCE_W//4), dtype=np.uint8)
    full_corners = REF_CORNERS + np.array([round(SOURCE_W*.32), 0], np.float32)
    cv2.fillConvexPoly(board_mask, np.rint(full_corners/4).astype(np.int32), 255)
    ref_features = feature_data(reference, board_mask)
    preview_dir = out / "alignment_previews"
    preview_dir.mkdir(exist_ok=True)
    missing = [base / relative for relative in WEIGHTS.values()
               if not (base / relative).is_file()]
    if missing and not args.alignment_only:
        raise FileNotFoundError(f"Missing weights: {missing}")
    models = {} if args.alignment_only else {name: YOLO(str(base / relative)) for name, relative in WEIGHTS.items()}

    sources = sorted(p for p in (base / "processed").rglob("*_4056x3040.jpg")
                     if p.name[:10] in days and p.name[11:13].isdigit()
                     and 8 <= int(p.name[11:13]) <= 18)
    if not sources:
        raise RuntimeError(f"No daytime 4056x3040 captures for {sorted(days)}")
    if args.limit:
        sources = sources[:args.limit]
    records, rejections, alignment_rows, cell_rows = [], [], [], []
    warmed = set()
    for index, source_path in enumerate(sources, 1):
        source = cv2.imread(str(source_path))
        if source is None or source.shape[:2] != (SOURCE_H, SOURCE_W):
            rejections.append(dict(capture=source_path.name, resolution="source",
                                   reason="unreadable or wrong dimensions"))
            continue
        matrix, quality, method, inliers = align(source, template, ref_features, args.min_match)
        alignment_rows.append(dict(capture=source_path.parent.name, method=method,
                                   template_quality=round(quality,4), inliers=inliers,
                                   accepted=int(matrix is not None)))
        if matrix is None:
            rejections.append(dict(capture=source_path.name, resolution="source",
                                   reason=f"{method}; template quality {quality:.4f}"))
            continue
        for resolution, (width, height) in RESOLUTIONS.items():
            candidates = sorted(source_path.parent.glob(f"*_{resolution}.jpg"))
            if len(candidates) != 1:
                rejections.append(dict(capture=source_path.parent.name, resolution=resolution,
                                       reason=f"expected one image, found {len(candidates)}"))
                continue
            image_path = candidates[0]
            frame = source if resolution == "4056x3040" else cv2.imread(str(image_path))
            if frame is None or frame.shape[:2] != (height, width):
                rejections.append(dict(capture=image_path.name, resolution=resolution,
                                       reason="unreadable or wrong dimensions"))
                continue
            sx, sy = width / SOURCE_W, height / SOURCE_H
            shifted = cv2.perspectiveTransform(full_corners[None], matrix)[0]
            shifted *= np.array([sx, sy], np.float32)
            if (np.any(shifted[:,0]<0) or np.any(shifted[:,0]>=width)
                    or np.any(shifted[:,1]<0) or np.any(shifted[:,1]>=height)):
                rejections.append(dict(capture=image_path.name, resolution=resolution,
                                       reason="board extends outside image"))
                continue
            if method == "template":
                # Preserve the reference crop, translating it with the camera.
                dx, dy = matrix[:2,2]
                left = round((round(SOURCE_W*.32)+dx)*sx)
                right = round((round(SOURCE_W*.70)+dx)*sx)
                top = max(0, round(dy*sy))
                bottom = min(height, round((round(SOURCE_H*.65)+dy)*sy))
            else:
                low, high = shifted.min(axis=0), shifted.max(axis=0)
                padding = (high-low)*np.array([.52,.14])
                left, top = np.floor(low-padding).astype(int)
                right, bottom = np.ceil(high+padding).astype(int)
            left, right = max(0,left), min(width,right)
            top, bottom = max(0,top), min(height,bottom)
            roi = frame[top:bottom, left:right]
            corners = shifted - np.array([left,top], np.float32)
            to_grid = cv2.getPerspectiveTransform(corners.astype(np.float32), GRID)
            if (args.alignment_only or index == 1 or method == "features"
                    or index % max(1,args.preview_every)==0):
                grid_preview(roi, corners, preview_dir /
                             f"{source_path.parent.name}_{resolution}.jpg")
            if args.alignment_only:
                continue
            for model_name, model in models.items():
                if (model_name,resolution) not in warmed:
                    model.predict(roi, imgsz=args.imgsz, conf=.25, iou=.45,
                                  rect=True, max_det=300, device=args.device, verbose=False)
                    warmed.add((model_name,resolution))
                start = time.perf_counter()
                prediction = model.predict(roi, imgsz=args.imgsz, conf=.25, iou=.45,
                                           rect=True, max_det=300, device=args.device, verbose=False)[0]
                elapsed_ms = (time.perf_counter() - start) * 1000
                best = {}
                outside = 0
                for box in prediction.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center = np.array([[[(x1 + x2) / 2, (y1 + y2) / 2]]], dtype=np.float32)
                    gx, gy = cv2.perspectiveTransform(center, to_grid)[0, 0]
                    if not (0 <= gx < 500 and 0 <= gy < 800):
                        outside += 1
                        continue
                    cell = (min(int(gy / 100), 7), min(int(gx / 100), 4))
                    confidence = float(box.conf[0])
                    if cell not in best or confidence > best[cell][1]:
                        best[cell] = (int(box.cls[0]), confidence)
                correct = int(sum(digit == TRUTH[row, col]
                              for (row, col), (digit, _) in best.items()))
                for row in range(8):
                    for col in range(5):
                        detected = best.get((row,col))
                        cell_rows.append(dict(capture=source_path.parent.name,
                            model=model_name, resolution=resolution,
                            row=row+1, column=col+1, actual=int(TRUTH[row,col]),
                            predicted=detected[0] if detected else "MISS",
                            confidence=round(detected[1],6) if detected else "",
                            correct=int(bool(detected and detected[0]==TRUTH[row,col]))))
                records.append(dict(capture=source_path.parent.name, image=image_path.name,
                                    date=image_path.name[:10], resolution=resolution,
                                    model=model_name, match_quality=round(quality, 4),
                                    alignment_method=method, feature_inliers=inliers,
                                    shift_x=round(float(matrix[0,2]),2), shift_y=round(float(matrix[1,2]),2), found=len(best), correct=correct,
                                    wrong=len(best)-correct, empty=40-len(best),
                                    predictions=len(prediction.boxes),
                                    extra_on_board=len(prediction.boxes)-len(best)-outside,
                                    outside=outside, meets_34_of_40=int(correct >= 34),
                                    elapsed_ms=round(elapsed_ms, 2)))
        if index % 10 == 0 or index == len(sources):
            print(f"Processed {index}/{len(sources)} capture folders", flush=True)

    write_rows(out / "alignment.csv", alignment_rows,
               ["capture","method","template_quality","inliers","accepted"])
    write_rows(out / "rejections.csv", rejections, ["capture","resolution","reason"])
    if args.alignment_only:
        print(f"Alignment review ready: {preview_dir}", flush=True)
        print(f"Accepted alignment: {sum(r['accepted'] for r in alignment_rows)}/{len(sources)}")
        return
    if not records:
        raise RuntimeError(f"No images scored; first rejections: {rejections[:5]}")
    # Compare configurations on identical capture folders, even if one
    # resolution was unreadable or its aligned crop fell outside the frame.
    by_capture = defaultdict(list)
    for record in records:
        by_capture[record["capture"]].append(record)
    complete = {capture for capture, rows in by_capture.items()
                if len(rows) == len(WEIGHTS) * len(RESOLUTIONS)}
    for capture, rows in by_capture.items():
        if capture not in complete:
            rejections.append(dict(capture=capture, resolution="all",
                                   reason=f"incomplete paired capture: {len(rows)}/12 scores"))
    records = [record for record in records if record["capture"] in complete]
    if not records:
        raise RuntimeError(f"No complete 12-configuration captures; rejections: {rejections[:8]}")
    write_rows(out / "per_image_scores.csv", records, list(records[0]))
    cell_rows = [r for r in cell_rows if r["capture"] in complete]
    write_rows(out / "per_digit_scores.csv", cell_rows, list(cell_rows[0]))
    write_rows(out / "rejections.csv", rejections,
               ["capture", "resolution", "reason"])
    summary = []
    for (date, model, resolution) in sorted({(r["date"], r["model"], r["resolution"])
                                            for r in records}):
        group = [r for r in records if (r["date"], r["model"], r["resolution"])
                 == (date, model, resolution)]
        summary.append(dict(date=date, model=model, resolution=resolution, images=len(group),
                            mean_found=round(statistics.mean(r["found"] for r in group), 3),
                            mean_correct=round(statistics.mean(r["correct"] for r in group), 3),
                            digit_accuracy=round(sum(r["correct"] for r in group)
                                                 / (40 * len(group)), 6),
                            boards_passing=sum(r["meets_34_of_40"] for r in group),
                            pass_rate=round(statistics.mean(r["meets_34_of_40"]
                                                            for r in group), 6),
                            mean_elapsed_ms=round(statistics.mean(r["elapsed_ms"]
                                                                  for r in group), 2)))
    write_rows(out / "per_day_summary.csv", summary, list(summary[0]))
    overall = []
    for model in WEIGHTS:
        for resolution in RESOLUTIONS:
            group = [r for r in records if r["model"] == model
                     and r["resolution"] == resolution]
            if not group:
                continue
            overall.append(dict(model=model, resolution=resolution, images=len(group),
                                days=len({r["date"] for r in group}),
                                mean_found=round(statistics.mean(r["found"] for r in group), 3),
                                mean_correct=round(statistics.mean(r["correct"] for r in group), 3),
                                digit_accuracy=round(sum(r["correct"] for r in group)
                                                     / (40 * len(group)), 6),
                                boards_passing=sum(r["meets_34_of_40"] for r in group),
                                pass_rate=round(statistics.mean(r["meets_34_of_40"]
                                                                for r in group), 6),
                                mean_elapsed_ms=round(statistics.mean(r["elapsed_ms"]
                                                                      for r in group), 2)))
    write_rows(out / "overall_summary.csv", overall, list(overall[0]))
    print("\nModel                 Resolution   Images  Mean correct  Passes   Mean ms")
    for row in overall:
        print(f"{row['model']:21} {row['resolution']:11} {row['images']:6} "
              f"{row['mean_correct']:12.2f} {row['boards_passing']:3}/{row['images']:<3} "
              f"{row['mean_elapsed_ms']:8.1f}")
    print(f"\nAlignment/image rejections: {len(rejections)}. Results: {out}")


if __name__ == "__main__":
    main()
