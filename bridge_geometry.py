# -*- coding: utf-8 -*-
"""Bridge surface-defect geometry from an existing 5-class segmentation mask.

No model weights, depth estimator, or model training is needed for this step.
Image units are PIXELS unless a verified planar calibration is supplied.
Class ids: 0 background; 1 crack; 2 spalling; 3 corrosion; 4 exposed_rebar;
255 unobserved. Image coordinates: x right, y down. box_xyxy upper bound exclusive.

These are computational geometry estimates, not certified bridge measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from PIL import Image

CLASSES = {1: ("crack", "裂缝"), 2: ("spalling", "剥落"),
           3: ("corrosion", "可见锈蚀迹象"), 4: ("exposed_rebar", "钢筋外露")}
# Example filters to suppress tiny components. Tune and validate on held-out images.
DEFAULT_MIN_REGION_PX = {1: 5, 2: 20, 3: 20, 4: 20}


def _p(v: float | int | None, n: int = 4):
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), n)


def _roi_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _orientation_deg(points_xy: np.ndarray) -> tuple[float | None, float | None]:
    """Axial PCA orientation in image coords; 0: horizontal, 90: vertical."""
    if len(points_xy) < 3:
        return None, None
    cv = np.cov(np.asarray(points_xy, dtype=np.float64).T)
    vals, vecs = np.linalg.eigh(cv)
    if vals[-1] < 1e-8:
        return None, None
    axis = vecs[:, -1]
    direction = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 180.0
    elongation = float(vals[-1] / max(float(vals[-2]), 1e-8))
    # In roughly circular/isotropic components, a global direction is not meaningful.
    if elongation < 1.5:
        return None, _p(elongation, 2)
    return _p(direction, 2), _p(elongation, 2)


# ---------------------- V2 topology-aware crack metrics ----------------------
from heapq import heappop, heappush
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates


def _skeleton_graph(sk: np.ndarray):
    """8-adjacency with redundant corner diagonals suppressed.

    A diagonal joining two skeleton pixels is not counted when the pixels are
    already connected by an orthogonal skeleton step through the 2x2 corner.
    This removes many digital 3-cycle artifacts in a nominal centerline.
    """
    coords = [tuple(map(int, v)) for v in np.argwhere(sk)]
    index = {node: i for i, node in enumerate(coords)}
    neighbors = [dict() for _ in coords]
    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                node = (y + dy, x + dx)
                j = index.get(node)
                if j is None or j <= i:
                    continue
                if dx and dy:
                    if (y, x + dx) in index or (y + dy, x) in index:
                        continue
                length = math.hypot(dx, dy)
                neighbors[i][j] = length
                neighbors[j][i] = length
    return coords, neighbors


def _border_touch(mask: np.ndarray):
    return {
        'left': bool(np.any(mask[:, 0])), 'right': bool(np.any(mask[:, -1])),
        'top': bool(np.any(mask[0, :])), 'bottom': bool(np.any(mask[-1, :])),
    }


def _prune_spurs(sk: np.ndarray, threshold_px: float = 12.0):
    """Remove short endpoint-to-junction side twigs; preserve clipped edge tips.

    Important: only algorithmic nuisance removal, not a verified correction for
    every genuine crack spur. Raw and refined lengths remain separately stored.
    """
    coords, adj = _skeleton_graph(sk)
    active = set(range(len(coords)))
    deleted_paths = 0
    while True:
        deg = {i: sum(j in active for j in adj[i]) for i in active}
        endpoints = [i for i in active if deg[i] == 1]
        removed_any = False
        for tip in endpoints:
            if tip not in active:
                continue
            y, x = coords[tip]
            if x <= 1 or y <= 1 or x >= sk.shape[1]-2 or y >= sk.shape[0]-2:
                continue
            path = [tip]
            prev = -1
            node = tip
            distance = 0.0
            while True:
                nxt = [(j, step) for j, step in adj[node].items() if j in active and j != prev]
                if len(nxt) != 1:
                    break
                new, step = nxt[0]
                distance += step
                if distance >= threshold_px:
                    break
                if deg.get(new, 0) >= 3:
                    # prune only if it terminates at a proper junction
                    if len(path) > 0:
                        for k in path:
                            active.discard(k)
                        deleted_paths += 1
                        removed_any = True
                    break
                if deg.get(new, 0) != 2:
                    break
                path.append(new)
                prev, node = node, new
            if removed_any:
                break
        if not removed_any:
            break
    refined = np.zeros_like(sk, dtype=bool)
    if active:
        ys, xs = np.array([coords[i] for i in active]).T
        refined[ys, xs] = True
    return refined, deleted_paths


def _graph_total_length(neighbors):
    return float(sum(length for row in neighbors for length in row.values()) / 2)


def _longest_endpoint_path(coords, graph):
    """Max weighted geodesic between endpoint candidates, not external crack extent."""
    endpoints = [i for i, edges in enumerate(graph) if len(edges) == 1]
    if len(endpoints) < 2:
        return None, []
    if len(endpoints) > 100:
        return None, []
    best_len, best_route = 0.0, []
    for start in endpoints:
        dist = {start: 0.0}
        prev = {}
        heap = [(0.0, start)]
        while heap:
            d, i = heappop(heap)
            if d != dist[i]:
                continue
            for j, w in graph[i].items():
                alt = d + w
                if alt < dist.get(j, float('inf')):
                    dist[j] = alt
                    prev[j] = i
                    heappush(heap, (alt, j))
        for target in endpoints:
            if target not in dist or dist[target] <= best_len:
                continue
            best_len = dist[target]
            route = [target]
            while route[-1] != start:
                route.append(prev[route[-1]])
            best_route = list(reversed(route))
    return round(best_len, 2), [coords[i] for i in best_route]


def _normal_width_sampling(mask, skel, graph_coords, graph,
                           min_junction_distance=None,
                           endpoint_distance=7.0, local_radius=14.0):
    """Estimated transverse mask width via centerline PCA tangents.

    Conservative sampling excludes junctions, endpoints, ambiguous tangents,
    and rays leaving the image. This is independent of physical calibration.
    """
    if not graph_coords:
        return {'count': 0, 'rejected': 0, 'summary': None}, [], np.empty((0,2))
    xy = np.array([(x, y) for y,x in graph_coords], dtype=np.float64)
    tree = cKDTree(xy)
    junction_xy = xy[[i for i,v in enumerate(graph) if len(v) >= 3]]
    endpoint_xy = xy[[i for i,v in enumerate(graph) if len(v) <= 1]]
    junction_tree = cKDTree(junction_xy) if len(junction_xy) else None
    endpoint_tree = cKDTree(endpoint_xy) if len(endpoint_xy) else None
    dist = distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
    # Exclude around junctions at a radius comparable to local crack width.
    if min_junction_distance is None:
        min_junction_distance = float(np.clip(np.median(2*dist[skel]) if np.any(skel) else 10, 10, 40))
    mask_float = mask.astype(np.float32)
    h, w = mask.shape
    samples = []
    rejected = {'endpoint_proximity':0,'junction_proximity':0,'border_proximity':0,
                'unstable_tangent':0,'open_boundary_or_ray':0,'disagrees_with_edt':0}
    # Dense samples for width percentiles; only draw a sparse subset on visualization.
    for idx,(x,y) in enumerate(xy):
        if len(graph[idx]) != 2:
            continue
        if x < 2 or y < 2 or x > w-3 or y > h-3:
            rejected['border_proximity'] += 1; continue
        if junction_tree is not None and junction_tree.query([x,y])[0] < min_junction_distance:
            rejected['junction_proximity'] += 1; continue
        if endpoint_tree is not None and endpoint_tree.query([x,y])[0] < endpoint_distance:
            rejected['endpoint_proximity'] += 1; continue
        ids = tree.query_ball_point([x,y], local_radius)
        if len(ids) < 7:
            rejected['unstable_tangent'] += 1; continue
        local = xy[ids]
        eigvals, eigvecs = np.linalg.eigh(np.cov(local.T))
        if eigvals[-1] < 1e-5 or eigvals[-1] / max(eigvals[0],1e-5) < 4:
            rejected['unstable_tangent'] += 1; continue
        tangent = eigvecs[:, -1]
        normal = np.array([-tangent[1], tangent[0]])
        # Subpixel ray march; leave at first mask boundary.
        edges = []
        valid = True
        for sign in (-1,1):
            hit = None
            for d in np.arange(0.25, 91.0, 0.5):
                xx, yy = x + normal[0]*d*sign, y + normal[1]*d*sign
                if xx < 0 or xx > w-1 or yy < 0 or yy > h-1:
                    valid = False; break
                value = float(map_coordinates(mask_float,
                         [[yy],[xx]],order=1,mode='constant',cval=0)[0])
                if value < 0.5:
                    hit = (xx,yy,d); break
            if hit is None:
                valid = False
            edges.append(hit)
            if not valid:
                break
        if not valid:
            rejected['open_boundary_or_ray'] += 1; continue
        measured = edges[0][2] + edges[1][2]
        edt_width = 2*dist[int(y),int(x)]
        if not (0.52 * edt_width <= measured <= 1.8 * edt_width):
            rejected['disagrees_with_edt'] += 1; continue
        samples.append({'x':int(x),'y':int(y),'width_px':round(measured,3),
                        'edt_px':round(float(edt_width),3),
                        'start_xy': [round(edges[0][0],2), round(edges[0][1],2)],
                        'end_xy': [round(edges[1][0],2), round(edges[1][1],2)]})
    vals = np.array([q['width_px'] for q in samples],dtype=float)
    summary = {'mean':_p(vals.mean(),3), 'median':_p(np.median(vals),3),
               'p95':_p(np.percentile(vals,95),3),'max':_p(vals.max(),3)} if len(vals) else None
    edt_paired = np.array([q['edt_px'] for q in samples], dtype=float)
    paired = None
    if len(vals):
        diff = vals - edt_paired
        paired = {
            'sample_count': len(vals),
            'edt_at_same_sample_points_px': {
                'mean':_p(edt_paired.mean(),3),
                'median':_p(np.median(edt_paired),3),
                'p95':_p(np.percentile(edt_paired,95),3),
                'max':_p(edt_paired.max(),3),
            },
            'normal_minus_edt_mean_px': _p(diff.mean(),3),
            'mean_absolute_disagreement_px': _p(np.abs(diff).mean(),3),
            'note': '同位置算法差异，不是相对于真实测量的误差，不表示任何一种方法更准确。',
        }
    return {'count':len(vals),'summary':summary,'rejection_counts':rejected,
            'refined_centerline_sample_denominator':len(xy),
            'coverage_fraction_of_centerline':_p(len(vals)/len(xy),4),
            'paired_width_comparison_same_points':paired,
            'junction_exclusion_radius_px':_p(min_junction_distance,2)}, samples, xy


def _crack_metrics(mask: np.ndarray) -> dict[str, Any]:
    sk_raw = skeletonize(mask.astype(bool))
    raw_coords, raw_graph = _skeleton_graph(sk_raw)
    sk, pruned_count = _prune_spurs(sk_raw, threshold_px=12)
    coords, graph = _skeleton_graph(sk)
    edt = distance_transform_edt(np.pad(mask,1,constant_values=False))[1:-1,1:-1]
    edt_vals = 2*edt[sk]
    ys, xs = np.where(sk)
    angle, elongation = _orientation_deg(np.column_stack([xs,ys]))
    branch_pixels = np.zeros(mask.shape,dtype=bool)
    for i, (y,x) in enumerate(coords):
        if len(graph[i]) >= 3:
            branch_pixels[y,x] = True
    branch_groups = cv2.connectedComponentsWithStats(branch_pixels.astype('uint8'),connectivity=8)[0]-1
    longest, route = _longest_endpoint_path(coords,graph)
    normal_stat, width_samples, _ = _normal_width_sampling(mask,sk,coords,graph)
    borders = _border_touch(mask)
    return {
        'skeleton_pixels':int(sk_raw.sum()),
        'network_skeleton_length_px_approx':_p(_graph_total_length(raw_graph),2),
        'width_px_edt_approx':{'mean':_p(edt_vals.mean(),3) if edt_vals.size else None,
                              'median':_p(np.median(edt_vals),3) if edt_vals.size else None,
                              'p95':_p(np.percentile(edt_vals,95),3) if edt_vals.size else None,
                              'max':_p(edt_vals.max(),3) if edt_vals.size else None},
        'main_axis_angle_deg_image':angle,
        'pca_elongation_ratio':elongation,
        'skeleton_endpoint_candidates':sum(len(x)==1 for x in graph),
        'skeleton_branch_candidates':sum(len(x)>=3 for x in graph),
        'refined_topology':{
           'raw_graph_network_length_px_approx':_p(_graph_total_length(raw_graph),2),
           'pruned_graph_network_length_px_approx':_p(_graph_total_length(graph),2),
           'removed_short_spur_count':pruned_count,
           'endpoint_count_after_pruning':sum(len(x)==1 for x in graph),
           'junction_pixel_count_after_pruning':int(branch_pixels.sum()),
           'junction_cluster_count_approx':int(branch_groups),
           'longest_endpoint_geodesic_length_px_approx':longest,
           'line_touches_crop_borders':borders,
           'visible_only_not_entire_crack':bool(any(borders.values())),
        },
        'width_px_normal_approx':normal_stat,
        'quality_flags':{
            'crop_truncated':bool(any(borders.values())),
            'max_edt_width_may_be_biased_near_bends':True,
            'width_requires_scale_for_mm':True,
        },
        'cautions':'骨架网络长度不一定等于主裂缝长度；裁剪边界处裂缝被截断；宽度均为像素估计，无真实尺度。'
    }


def crack_diagnostic_layers(mask: np.ndarray):
    """V2 visualization data (not serialized as large arrays in report)."""
    sk_raw = skeletonize(mask.astype(bool))
    sk, _ = _prune_spurs(sk_raw,12)
    coords, graph = _skeleton_graph(sk)
    _, samples, _ = _normal_width_sampling(mask, sk,coords,graph)
    longest, route = _longest_endpoint_path(coords,graph)
    return sk, samples, route


def _cc_components(class_mask: np.ndarray, min_region_pixels: int):
    n, comp, stats, centers = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(class_mask, dtype=np.uint8), connectivity=8
    )
    keep = np.zeros(class_mask.shape, dtype=bool)
    components = []
    for k in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[k]]
        if area < min_region_pixels:
            continue
        local = comp[y:y+h, x:x+w] == k
        keep[y:y+h, x:x+w] |= local
        components.append((x, y, w, h, area, local, [float(centers[k][0]), float(centers[k][1])]))
    return keep, components


def quantify_mask(
    labels: np.ndarray,
    structure_roi: np.ndarray | None = None,
    mm_per_pixel: float | None = None,
    calibration_reference: str | None = None,
    plane_rectified: bool = False,
    min_region_pixels: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Quantify semantic segmentation; no ground truth needed.

    `structure_roi` is a manually provided structural-element ROI, not the damage
    mask. Without it, area fractions refer only to the observed IMAGE region.
    Calibrated measurements require valid scale AND rectified planar coordinates.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a 2D integer class-ID mask")
    if not np.isin(labels, [0, 1, 2, 3, 4, 255]).all():
        raise ValueError("Unexpected label ID: expected 0..4 and 255 (unobserved)")
    if mm_per_pixel is not None:
        if not (np.isfinite(mm_per_pixel) and mm_per_pixel > 0):
            raise ValueError("mm_per_pixel must be a positive finite number")
        if not plane_rectified or not calibration_reference:
            raise ValueError("Physical units require --plane-rectified and --calibration-reference")
    if structure_roi is not None:
        if structure_roi.shape != labels.shape:
            raise ValueError("structure_roi and labels must share HxW coordinates")
        structure_roi = np.asarray(structure_roi) > 0

    observed = labels != 255
    effective_roi = observed if structure_roi is None else (observed & structure_roi)
    denominator = int(effective_roi.sum())
    if denominator == 0:
        raise ValueError("No observed pixels inside requested ROI")
    h, w = labels.shape
    source = "structure_roi_intersection_observed" if structure_roi is not None else "center_crop_observed_pixels_not_component_area"
    filters = DEFAULT_MIN_REGION_PX | (min_region_pixels or {})
    classes: list[dict[str, Any]] = []
    for class_id, (code, name) in CLASSES.items():
        raw = (labels == class_id) & effective_roi
        clean, components = _cc_components(raw, int(filters[class_id]))
        total = int(clean.sum())
        region_details = []
        for region_no, (x, y, rw, rh, area, local_mask, center) in enumerate(components, start=1):
            yy, xx = np.where(local_mask)
            points_xy = np.column_stack([xx + x, yy + y])
            angle, elongation = _orientation_deg(points_xy)
            # Contours define a morphological perimeter in image pixel coordinates.
            contours, _ = cv2.findContours(local_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
            # Rotated rectangles describe maximal extents, not material thickness.
            rect = cv2.minAreaRect(points_xy.astype(np.float32)) if len(points_xy) >= 3 else None
            geom = {
                "region_id": f"{code}_{region_no:04d}",
                "pixel_area": area,
                "area_fraction_of_observed_roi": _p(area / denominator, 7),
                "bbox_xyxy_exclusive": [x, y, x+rw, y+rh],
                "centroid_xy": [_p(center[0], 2), _p(center[1], 2)],
                "perimeter_px_approx": _p(perimeter, 2),
                "rotated_box_dimensions_px_approx": sorted([_p(z, 2) for z in rect[1]], reverse=True) if rect else None,
                "main_axis_angle_deg_image": angle,
                "pca_elongation_ratio": elongation,
                "touches_image_borders": {"left":x==0,"top":y==0,"right":x+rw==w,"bottom":y+rh==h},
            }
            if class_id == 1:
                # Quantify in FULL image coordinates.  Using a component's tight
                # bounding box makes every side look like an image edge and
                # biases width at the artificial crop boundaries.
                full_component = np.zeros((h, w), dtype=bool)
                full_component[y:y+rh, x:x+rw] = local_mask
                geom["crack_geometry"] = _crack_metrics(full_component)
            if class_id == 4:
                # Exposed rebar can be elongated, but min-area box is NOT proof
                # of actual rebar visible length. Do not equate the two.
                geom["visible_rebar_extent_method"] = "投影区域旋转外接矩形；不代表经验证的钢筋实际外露长度"
            if mm_per_pixel is not None:
                physical = {
                    "projected_area_mm2": _p(area * mm_per_pixel**2, 3),
                    "projected_perimeter_mm_approx": _p(perimeter * mm_per_pixel, 3),
                }
                if class_id == 1:
                    cg = geom["crack_geometry"]
                    physical["crack_network_length_mm_approx"] = _p(cg["network_skeleton_length_px_approx"] * mm_per_pixel, 3)
                    physical["crack_width_mm_edt_approx"] = {k: _p(v * mm_per_pixel, 4) if v is not None else None
                        for k, v in cg["width_px_edt_approx"].items()}
                geom["calibrated_projected_measurements"] = physical
            region_details.append(geom)
        class_bbox = _roi_bbox(clean)
        center_xy = None
        zone = None
        if class_bbox is not None:
            ys, xs = np.where(clean)
            cx, cy = float(xs.mean()), float(ys.mean())
            center_xy = [_p(cx, 2), _p(cy, 2)]
            zone = {
                "row": ("top", "middle", "bottom")[min(2, int(cy * 3 / h))],
                "col": ("left", "center", "right")[min(2, int(cx * 3 / w))],
            }
        classes.append({
            "class_id": class_id, "code": code, "name": name,
            "raw_predicted_pixels_in_roi": int(raw.sum()),
            "retained_pixels_after_filter": total,
            "retained_area_fraction": _p(total / denominator, 7),
            "retained_area_percent": _p(100 * total / denominator, 4),
            "candidate_connected_region_count": len(components),
            "distribution_bbox_xyxy_exclusive": class_bbox,
            "touches_image_borders": _border_touch(clean),
            "distribution_centroid_xy": center_xy,
            "distribution_3x3_grid_position": zone,
            "crack_overall_geometry": _crack_metrics(clean) if class_id == 1 and total else None,
            "calibrated_projected_area_mm2": _p(total * mm_per_pixel**2, 3) if mm_per_pixel is not None else None,
            "regions": region_details,
        })
    return {
        "schema_version": "2.1",
        "model_class_id_map": {str(k): {"code": v[0], "name": v[1]} for k,v in CLASSES.items()},
        "image_size_wh": [int(w), int(h)],
        "observed_pixel_count": int(observed.sum()),
        "observed_fraction_of_image": _p(observed.mean(), 6),
        "area_ratio_denominator": source,
        "roi_observed_pixel_count": denominator,
        "roi_bbox_xyxy_exclusive": _roi_bbox(effective_roi),
        "calibration": {
            "physical_units_available": mm_per_pixel is not None,
            "mm_per_pixel": mm_per_pixel,
            "reference": calibration_reference,
            "plane_rectified": bool(plane_rectified),
            "caution": "物理单位需要核实相机标定、标尺及共面透视校正；桥梁曲面不可直接用统一像素尺度。"
        },
        "connected_component_semantics": "候选连通域数，不等于真实独立病害数量",
        "threshold_min_region_pixels": {str(k): int(v) for k,v in filters.items()},
        "classes": classes,
        "scope_limitations": [
            "仅评估图像可观察表面，不代表内部病害或真实材料损失",
            "锈蚀指标为可见锈蚀迹象投影，不是钢筋截面损失或腐蚀程度",
            "剥落面积为可见表面投影，不代表材料体积损失",
            "未输入结构构件ROI时，面积占比相对可观察图像范围而非桥梁构件表面积",
            "像素分割的边缘误差、网络分叉和透视均会影响细长病害几何量化",
        ],
    }



def main():
    parser = argparse.ArgumentParser(description="Bridge damage geometry from 5-class class-ID mask (no depth model)")
    parser.add_argument("--mask", required=True, help="Path to grayscale 0..4,255 class-ID PNG")
    parser.add_argument("--output", required=True, help="JSON report output path")
    parser.add_argument("--roi-mask", default=None, help="Binary ROI of bridge structural element")
    parser.add_argument("--mm-per-pixel", type=float, default=None)
    parser.add_argument("--calibration-reference", default=None)
    parser.add_argument("--plane-rectified", action="store_true")
    args = parser.parse_args()
    labels = np.asarray(Image.open(args.mask))
    roi = np.asarray(Image.open(args.roi_mask).convert("L")) > 0 if args.roi_mask else None
    result = quantify_mask(labels, roi, args.mm_per_pixel, args.calibration_reference, args.plane_rectified)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"quantification_json": str(output), "observed_pixels": result["observed_pixel_count"],
        "regions": {c["code"]: c["candidate_connected_region_count"] for c in result["classes"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
