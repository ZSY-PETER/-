# -*- coding: utf-8 -*-
"""HRCDS bridge surface-defect recognition/segmentation + PIXEL geometry quantification.

No YOLO, ZoeDepth, RAG, retraining, or scale hallucination.
No arguments: infer Desktop/HRCDS/test_image/test_0001.jpg using the already verified
Desktop/best_model.pth + Desktop/VMamba-main official source.
--mask ...: skip model loading entirely, quantify an existing grayscale class-id mask.
"""
from __future__ import annotations
from bridge_paths import default_checkpoint, default_vmamba_models

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from utf8_process import set_utf8_stdio
from bridge_geometry import quantify_mask, CLASSES, DEFAULT_MIN_REGION_PX, crack_diagnostic_layers
from bridge_model import infer, colorize
from presentation_groups import grouped_quantification

COLORS = {0: (0, 0, 0), 1: (230, 35, 35), 2: (255, 140, 0),
          3: (230, 215, 0), 4: (0, 200, 230)}
HOME = Path.home()


def read_integer_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in {"L", "I", "I;16", "P"}:
            raise ValueError(f"Expected single-channel class-ID mask, not color/overlay: mode={image.mode}")
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")
    if not np.isin(mask, [0, 1, 2, 3, 4, 255]).all():
        vals = np.unique(mask)
        raise ValueError(f"Mask contains invalid class IDs {vals[:25]}; use the grayscale class-ID PNG, not a visualization")
    return mask.astype(np.uint8)


def fit_center_image(image_path: Path, wh: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Display-time alignment only. Never resize to claim physical dimensions."""
    with Image.open(image_path) as im:
        source = im.convert("RGB")
    sw, sh = source.size
    w, h = wh
    if (sw, sh) == (w, h):
        return np.asarray(source).copy(), {"original_wh": [sw, sh], "crop_xy": [0, 0], "padding_right_bottom": [0, 0]}
    pr, pb = max(0, w - sw), max(0, h - sh)
    if pr or pb:
        source = ImageOps.expand(source, border=(0, 0, pr, pb), fill=0)
    x, y = (source.width - w)//2, (source.height - h)//2
    crop = source.crop((x, y, x+w, y+h))
    return np.asarray(crop).copy(), {"original_wh": [sw, sh], "crop_xy": [x, y], "padding_right_bottom": [pr, pb]}


def write_image(path: Path, array: np.ndarray):
    Image.fromarray(array).save(path)


def mask_from_filtered_regions(labels: np.ndarray, report: dict) -> np.ndarray:
    kept = np.zeros_like(labels, dtype=np.uint8)
    for cl in report["classes"]:
        idx = cl["class_id"]
        filtered = np.zeros(labels.shape, dtype=np.uint8)
        n, cc, stats, centers = cv2.connectedComponentsWithStats(
            (labels == idx).astype(np.uint8), connectivity=8)
        minimum = report["threshold_min_region_pixels"][str(idx)]
        for component_id in range(1, n):
            if stats[component_id, cv2.CC_STAT_AREA] >= minimum:
                filtered[cc == component_id] = idx
        kept[filtered == idx] = idx
    kept[labels == 255] = 255
    return kept


def render_outputs(labels: np.ndarray, original: np.ndarray | None, report: dict, out: Path):
    h, w = labels.shape
    # Keep native five-class output intact; produce four-class display separately.
    color_mask = colorize(labels)
    display_labels = labels.copy()
    display_labels[display_labels == 4] = 3  # steel-related damage display only
    display_color_mask = colorize(display_labels)
    if original is None:
        original = np.full((h, w, 3), 205, dtype=np.uint8)
    observed_damage = (labels > 0) & (labels < 5)
    overlay_native = original.copy()
    overlay_native[observed_damage] = np.clip(
        original[observed_damage].astype(np.float32)*0.48 +
        color_mask[observed_damage].astype(np.float32)*0.52,
        0, 255).astype(np.uint8)
    overlay = original.copy()
    overlay[observed_damage] = np.clip(
        original[observed_damage].astype(np.float32)*0.48 +
        display_color_mask[observed_damage].astype(np.float32)*0.52,
        0, 255).astype(np.uint8)
    annotated = overlay.copy()
    for idx in (1, 2, 3):
        # Contours are merged only for display; the five-class mask is unchanged.
        u8 = (display_labels == idx).astype(np.uint8)
        contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, COLORS[idx], 2)

    write_image(out / "01_center_crop.png", original)
    write_image(out / "02_class_ids.png", labels)
    write_image(out / "03_color_mask.png", color_mask)
    write_image(out / "03_display_merged_mask.png", display_color_mask)
    write_image(out / "04_overlay.png", overlay)
    write_image(out / "04_overlay_five_class.png", overlay_native)
    write_image(out / "05_contours.png", annotated)
    combined = np.concatenate([original, display_color_mask, annotated], axis=1)
    write_image(out / "06_three_panel.png", combined)

    for idx, code in [(1, "crack"), (2, "spalling"), (3, "corrosion"), (4, "exposed_rebar")]:
        write_image(out / f"class_{idx}_{code}_binary.png", (labels == idx).astype(np.uint8) * 255)

    crack = (labels == 1)
    skel = skeletonize(crack)
    skel_pic = original.copy()
    skel_pic[crack] = (230, 35, 35)
    skel_pic[skel] = (15, 235, 70)
    write_image(out / "07_crack_skeleton.png", skel_pic)

    dist = distance_transform_edt(np.pad(crack, 1, constant_values=False))[1:-1, 1:-1]
    width_map = np.zeros((h, w), dtype=np.uint8)
    if np.any(skel):
        vals = 2*dist[skel]
        high = max(1.0, float(np.percentile(vals, 95)))
        width_map[skel] = np.clip((2*dist[skel] / high) * 255, 1, 255).astype(np.uint8)
    heatmap = cv2.cvtColor(cv2.applyColorMap(width_map, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    heatmap[~skel] = (0,0,0)
    write_image(out / "08_crack_width_skeleton_heatmap.png", heatmap)

    # V2 quality-control images: refined skeleton and sampled normals (sparse).
    clean_skel, width_samples, long_route = crack_diagnostic_layers(crack)
    qc = original.copy()
    qc[crack] = np.clip(qc[crack].astype(float)*0.50 + np.array((230,35,35))*0.50,0,255).astype(np.uint8)
    qc[clean_skel] = (25,235,75)
    for record in width_samples[::max(1, len(width_samples)//45)]:
        xy1 = tuple(int(round(t)) for t in record['start_xy'])
        xy2 = tuple(int(round(t)) for t in record['end_xy'])
        cv2.line(qc, xy1, xy2, (40,210,245), 1, cv2.LINE_AA)
    write_image(out / "11_crack_normal_width_samples.png", qc)
    route_img = original.copy()
    route_img[crack] = np.clip(route_img[crack].astype(float)*0.65 + np.array((230,35,35))*0.35,0,255).astype(np.uint8)
    route_img[clean_skel] = (50,225,90)
    if len(long_route) > 1:
        pts = np.array([(x,y) for y,x in long_route], dtype=np.int32)
        cv2.polylines(route_img, [pts], False, (255,210,40), 2, cv2.LINE_AA)
    write_image(out / "12_crack_longest_geodesic.png",route_img)


def save_csv(report: dict, path: Path):
    fields = ["class_id", "class_code", "class_name_zh", "raw_area_px2",
              "area_px2_after_filter", "area_percent_of_observed_roi",
              "candidate_regions", "overall_bbox_xyxy_exclusive", "centroid_xy",
              "crack_network_length_px_approx", "crack_mean_width_px_edt_approx",
              "crack_p95_width_px_edt_approx", "crack_max_width_px_edt_approx",
              "crack_main_axis_deg_image",
              "refined_crack_network_length_px_approx", "longest_endpoint_path_px_approx",
              "width_normal_mean_px_approx", "width_normal_p95_px_approx",
              "width_normal_max_px_approx", "width_normal_sample_coverage", "touches_crop_border"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in report["classes"]:
            g = c.get("crack_overall_geometry") or {}
            width = g.get("width_px_edt_approx") or {}
            writer.writerow({
                "class_id": c["class_id"], "class_code": c["code"], "class_name_zh": c["name"],
                "raw_area_px2": c["raw_predicted_pixels_in_roi"],
                "area_px2_after_filter": c["retained_pixels_after_filter"],
                "area_percent_of_observed_roi": c["retained_area_percent"],
                "candidate_regions": c["candidate_connected_region_count"],
                "overall_bbox_xyxy_exclusive": json.dumps(c["distribution_bbox_xyxy_exclusive"]),
                "centroid_xy": json.dumps(c.get("distribution_centroid_xy")),
                "crack_network_length_px_approx": g.get("network_skeleton_length_px_approx"),
                "crack_mean_width_px_edt_approx": width.get("mean"),
                "crack_p95_width_px_edt_approx": width.get("p95"),
                "crack_max_width_px_edt_approx": width.get("max"),
                "crack_main_axis_deg_image": g.get("main_axis_angle_deg_image"),
                "refined_crack_network_length_px_approx": (g.get('refined_topology') or {}).get('pruned_graph_network_length_px_approx'),
                "longest_endpoint_path_px_approx": (g.get('refined_topology') or {}).get('longest_endpoint_geodesic_length_px_approx'),
                "width_normal_mean_px_approx": ((g.get('width_px_normal_approx') or {}).get('summary') or {}).get('mean'),
                "width_normal_p95_px_approx": ((g.get('width_px_normal_approx') or {}).get('summary') or {}).get('p95'),
                "width_normal_max_px_approx": ((g.get('width_px_normal_approx') or {}).get('summary') or {}).get('max'),
                "width_normal_sample_coverage": (g.get('width_px_normal_approx') or {}).get('coverage_fraction_of_centerline'),
                "touches_crop_border": int(any(c.get('touches_image_borders',{}).values())),
            })


def main():
    parser = argparse.ArgumentParser(description="BridgeDamageQuantifier: inference or existing-mask pixel geometry, NO ZoeDepth, NO RAG")
    parser.add_argument("--image", type=Path, default=None, help="Image path; no arguments use HRCDS test_0001.jpg")
    parser.add_argument("--mask", type=Path, default=None, help="Existing 0-4 grayscale mask; skips model inference")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--vmamba-models", type=Path, default=default_vmamba_models())
    parser.add_argument("--out", type=Path, default=HOME / "Desktop/BridgeDamageQuantifier_V2_1_Output")
    parser.add_argument("--roi", type=Path, default=None, help="Optional grayscale binary ROI, same shape as class mask")
    parser.add_argument("--min-crack-px", type=int, default=5)
    parser.add_argument("--min-region-px", type=int, default=20)
    args = parser.parse_args()
    if args.image is None and args.mask is None:
        args.image = HOME / "Desktop/HRCDS/test_image/test_0001.jpg"
    if args.min_crack_px < 1 or args.min_region_px < 1:
        parser.error("Component-size thresholds must be positive integers")

    if args.mask:
        if not args.mask.is_file():
            raise FileNotFoundError(f"Mask not found: {args.mask}")
        mask = read_integer_mask(args.mask)
        if args.image is not None:
            if not args.image.is_file():
                raise FileNotFoundError(f"Optional overlay image not found: {args.image}")
            image_rgb, crop_meta = fit_center_image(args.image, (mask.shape[1], mask.shape[0]))
        else:
            image_rgb, crop_meta = None, None
        metadata = {"mode": "existing_segmentation_mask", "mask_source": str(args.mask.resolve()),
                    "original_image": str(args.image) if args.image is not None else None,
                    "image_alignment": crop_meta}
    else:
        if not args.image.is_file():
            raise FileNotFoundError(f"Image not found: {args.image}")
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")
        if not (args.vmamba_models / "vmamba.py").is_file():
            raise FileNotFoundError(f"Official VMamba vmamba.py not found: {args.vmamba_models}")
        print("[1/3] Running DeepLabV3+ + official SS2D inference (640x480 center crop)...", flush=True)
        t = time.perf_counter()
        mask, image_rgb, model_info = infer(args.image, args.checkpoint, args.vmamba_models)
        print(f"Model inference done in {time.perf_counter()-t:.2f}s", flush=True)
        metadata = {"mode":"model_inference", "checkpoint": str(args.checkpoint.resolve()), **model_info}

    out = args.out / (args.image.stem if args.image is not None else args.mask.stem)
    out.mkdir(parents=True, exist_ok=True)
    roi = None
    if args.roi:
        if not args.roi.exists():
            raise FileNotFoundError(f"ROI file not found: {args.roi}")
        with Image.open(args.roi) as roi_img:
            roi = np.asarray(roi_img.convert("L")) > 0
        if roi.shape != mask.shape:
            raise ValueError(f"ROI {roi.shape} and mask {mask.shape} shape mismatch")
    print("[2/3] Quantifying mask geometry (pixel units only)...", flush=True)
    filters = {1: args.min_crack_px, 2: args.min_region_px, 3: args.min_region_px, 4: args.min_region_px}
    report = quantify_mask(mask, structure_roi=roi, min_region_pixels=filters)
    report["presentation_groups"] = grouped_quantification(report)
    report["input"] = metadata
    report["method"] = {
        "model_class_ids_are_from_HRCDS_GT": True,
        "class_id_note": "HRCDS GT: 0 background,1 crack,2 spalling,3 corrosion,4 exposed_rebar; ignores incorrect checkpoint config.class_names",
        "no_physical_scale": True,
        "unit_convention": "length/width px, visible projected area px2, fractions relative to observed crop or ROI",
        "predicted_or_gt": "predicted_mask_not_ground_truth" if not args.mask else "provided_existing_mask_origin_unspecified",
        "crop_scope": "model mode: 640x480 center crop; outside crop unobserved",
        "crack_width_formula": "EDT baseline + experimental local-normal width, paired comparison at identical retained centerline points; pixels only, not physically validated",
    }
    print("[3/3] Rendering visualization and writing JSON/CSV...", flush=True)
    render_outputs(mask, image_rgb, report, out)
    json_file = out / "09_quantification.json"
    json_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(report, out / "10_summary.csv")
    for c in report["classes"]:
        print(f"  ID{c['class_id']} {c['code']:<14} raw_px2={c['raw_predicted_pixels_in_roi']:>7} "
              f"filtered_px2={c['retained_pixels_after_filter']:>7} "
              f"area={c['retained_area_percent']:>8}% "
              f"regions={c['candidate_connected_region_count']}")
        if c["code"] == "crack" and c["crack_overall_geometry"]:
            geometry = c["crack_overall_geometry"]
            refined = geometry.get("refined_topology") or {}
            width = (geometry.get("width_px_normal_approx") or {}).get("summary") or {}
            print("     crack pixel geometry:"
                  f" visible_main_path_px={refined.get('longest_endpoint_geodesic_length_px_approx')}"
                  f" width_mean_px={width.get('mean')}"
                  f" width_max_px={width.get('max')}", flush=True)
    print("DONE output:", out, flush=True)

if __name__ == "__main__":
    set_utf8_stdio()
    main()
