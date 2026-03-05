import pdfplumber
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")          # ✅ add this BEFORE importing pyplot
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from typing import List, Tuple, Optional
from dataclasses import dataclass

Segment = Tuple[float, float, float, float]

@dataclass
class PdfEdgesMeta:
    page_width: float
    page_height: float
    bbox: Tuple[float, float, float, float]
    shift: Tuple[float, float]
    method: str

# ----------------------------
# Dimension removal via "ticks"
# ----------------------------

def _seg_len_ang(s: Segment) -> Tuple[float, float]:
    x1, y1, x2, y2 = s
    dx, dy = x2 - x1, y2 - y1
    L = float(np.hypot(dx, dy))
    ang = float(np.arctan2(dy, dx))  # radians
    return L, ang

def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 2*np.pi - d)

def _pt_dist(p, q) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))

def remove_dimension_by_ticks(
    segments: List[Segment],
    t_short: float = 12.0,       # "tick/arrow" segment max length (in PDF units)
    t_long: float = 80.0,        # dimension baseline min length (in PDF units)
    end_radius: float = 18.0,    # how close a tick must be to a long line end
    perp_tol_deg: float = 25.0,  # how close to perpendicular (deg)
) -> List[Segment]:
    """
    Removes likely dimension baselines using the pattern:
      long straight segment + short perpendicular tick near BOTH ends.
    Also removes the tick segments themselves (near removed baselines).
    """
    if not segments:
        return segments

    perp_tol = np.deg2rad(perp_tol_deg)

    info = []
    for s in segments:
        L, ang = _seg_len_ang(s)
        x1, y1, x2, y2 = s
        info.append((s, L, ang, (x1, y1), (x2, y2)))

    shorts = [(s, L, ang, p1, p2) for (s, L, ang, p1, p2) in info if L <= t_short]
    longs  = [(s, L, ang, p1, p2) for (s, L, ang, p1, p2) in info if L >= t_long]

    if not shorts or not longs:
        return segments

    dim_lines = set()

    # Identify long segments that have perpendicular ticks near both ends
    for (sL, LL, aL, e1, e2) in longs:
        hits1 = 0
        hits2 = 0

        for (sS, LS, aS, p1, p2) in shorts:
            # close to end 1?
            if (_pt_dist(p1, e1) < end_radius) or (_pt_dist(p2, e1) < end_radius):
                if abs(_angle_diff(aS, aL) - np.pi/2) < perp_tol:
                    hits1 += 1

            # close to end 2?
            if (_pt_dist(p1, e2) < end_radius) or (_pt_dist(p2, e2) < end_radius):
                if abs(_angle_diff(aS, aL) - np.pi/2) < perp_tol:
                    hits2 += 1

            if hits1 >= 1 and hits2 >= 1:
                break

        if hits1 >= 1 and hits2 >= 1:
            dim_lines.add(sL)

    if not dim_lines:
        return segments

    # Also remove short tick segments near the removed dimension baselines
    ticks_to_remove = set()
    for (sL, LL, aL, e1, e2) in [(s,L,ang,p1,p2) for (s,L,ang,p1,p2) in info if s in dim_lines]:
        for (sS, LS, aS, p1, p2) in shorts:
            near_end = (
                _pt_dist(p1, e1) < end_radius or _pt_dist(p2, e1) < end_radius or
                _pt_dist(p1, e2) < end_radius or _pt_dist(p2, e2) < end_radius
            )
            if near_end and abs(_angle_diff(aS, aL) - np.pi/2) < perp_tol:
                ticks_to_remove.add(sS)

    kept = [s for s in segments if (s not in dim_lines and s not in ticks_to_remove)]
    return kept

# ----------------------------
# Existing bbox logic
# ----------------------------

def get_pdf_content_bbox(segments: List[Segment]) -> Tuple[float, float, float, float]:
    """
    Identifies the building and fences by scoring clusters based on geometric complexity.
    This prevents large, empty frames or title blocks from being selected over the house.
    """
    if not segments:
        return (0, 0, 1, 1)

    # 1. PRE-FILTER: Ignore lines that are likely part of the page border/frame
    filtered_segs = []
    for x1, y1, x2, y2 in segments:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 650:  # Adjust based on typical page scale
            filtered_segs.append((x1, y1, x2, y2))

    if not filtered_segs:
        filtered_segs = segments

    # 2. Extract midpoints for clustering
    midpoints = np.array([[(s[0] + s[2]) / 2, (s[1] + s[3]) / 2] for s in filtered_segs])

    # 3. Cluster: 'eps' of 60 units bridges the gap between house and fences
    clustering = DBSCAN(eps=60, min_samples=3).fit(midpoints)
    labels = clustering.labels_

    unique_labels = set(labels) - {-1}  # Ignore noise label -1

    if not unique_labels:
        pts = np.array([[(s[0], s[1]), (s[2], s[3])] for s in filtered_segs]).reshape(-1, 2)
        return (np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 0]), np.max(pts[:, 1]))

    # 4. Score each cluster by Complexity Density AND Size
    best_label = -1
    highest_score = -1

    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]

        # Minimum line threshold
        if len(indices) < 15:
            continue

        cluster_pts = []
        for idx in indices:
            s = filtered_segs[idx]
            cluster_pts.extend([(s[0], s[1]), (s[2], s[3])])

        pts_arr = np.array(cluster_pts)
        w = np.max(pts_arr[:, 0]) - np.min(pts_arr[:, 0])
        h = np.max(pts_arr[:, 1]) - np.min(pts_arr[:, 1])
        area = (w * h) + 1.0

        avg_len = np.mean([
            np.sqrt((filtered_segs[i][2] - filtered_segs[i][0]) ** 2 +
                    (filtered_segs[i][3] - filtered_segs[i][1]) ** 2)
            for i in indices
        ])

        score = (len(indices) / (area / 1000.0)) * avg_len

        if score > highest_score:
            highest_score = score
            best_label = label

    final_indices = [i for i, l in enumerate(labels) if l == best_label]
    final_pts = []
    for idx in final_indices:
        s = filtered_segs[idx]
        final_pts.extend([(s[0], s[1]), (s[2], s[3])])

    pts_arr = np.array(final_pts)
    return (np.min(pts_arr[:, 0]), np.min(pts_arr[:, 1]),
            np.max(pts_arr[:, 0]), np.max(pts_arr[:, 1]))

# ----------------------------
# Public API
# ----------------------------

def extract_pdf_edges(pdf_path: str, page_num: int = 0) -> Tuple[List[Segment], PdfEdgesMeta]:
    segments, meta = _extract_vector_edges(pdf_path, page_num)
    # If vector data is missing/scanned, fall back to raster
    if len(segments) < 50:
        segments, meta = _extract_raster_edges(pdf_path, page_num)
    return segments, meta

def _extract_vector_edges(pdf_path: str, page_num: int) -> Tuple[List[Segment], PdfEdgesMeta]:
    segments: List[Segment] = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        width, height = float(page.width), float(page.height)

        for line in page.lines:
            x1, y1 = float(line['x0']), height - float(line['top'])
            x2, y2 = float(line['x1']), height - float(line['bottom'])
            segments.append((x1, y1, x2, y2))

        for rect in page.rects:
            x0 = float(rect['x0'])
            y0 = height - float(rect['bottom'])
            x1 = float(rect['x1'])
            y1 = height - float(rect['top'])
            segments.extend([
                (x0, y0, x1, y0),
                (x1, y0, x1, y1),
                (x1, y1, x0, y1),
                (x0, y1, x0, y0)
            ])

    # ✅ NEW: remove dimension-like lines using tick logic
    # You may need to tune t_short/t_long depending on your PDF units.
    segments = remove_dimension_by_ticks(segments, t_short=12.0, t_long=80.0, end_radius=18.0, perp_tol_deg=25.0)

    min_x, min_y, max_x, max_y = get_pdf_content_bbox(segments)

    # Apply final filtering: keep lines within reasonable distance of house cluster
    shifted = []
    padding = 80
    for x1, y1, x2, y2 in segments:
        if (min_x - padding <= x1 <= max_x + padding) and (min_y - padding <= y1 <= max_y + padding):
            shifted.append((x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y))

    meta = PdfEdgesMeta(
        page_width=width,
        page_height=height,
        bbox=(0, 0, max_x - min_x, max_y - min_y),
        shift=(min_x, min_y),
        method="vector"
    )
    return shifted, meta

def _extract_raster_edges(pdf_path: str, page_num: int) -> Tuple[List[Segment], PdfEdgesMeta]:
    import fitz
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=5)

    raw: List[Segment] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1, y1, x2, y2 = x1 / 2.0, y1 / 2.0, x2 / 2.0, y2 / 2.0
            ry1, ry2 = page.rect.height - y1, page.rect.height - y2
            raw.append((float(x1), float(ry1), float(x2), float(ry2)))

    # Note: tick-logic removal is vector-based; raster fallback stays as-is.
    min_x, min_y, max_x, max_y = get_pdf_content_bbox(raw)
    padding = 80
    shifted = [
        (x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y)
        for x1, y1, x2, y2 in raw
        if min_x - padding <= x1 <= max_x + padding
    ]

    return shifted, PdfEdgesMeta(
        page.rect.width,
        page.rect.height,
        (0, 0, max_x - min_x, max_y - min_y),
        (min_x, min_y),
        "raster"
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_edges.py <path_to_pdf>")
    else:
        path = sys.argv[1]
        try:
            segs, meta = extract_pdf_edges(path)

            plt.figure(figsize=(10, 10))
            for x1, y1, x2, y2 in segs:
                plt.plot([x1, x2], [y1, y2], color='blue', linewidth=0.5)

            plt.title(f"Complexity-Filtered PDF: {meta.bbox[2]:.2f}x{meta.bbox[3]:.2f} units")
            plt.axis('equal')
            plt.grid(True, linestyle='--', alpha=0.6)

            output_path = "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/working1floor/test.png"
            plt.savefig(output_path, dpi=200)
            plt.close()

            print(f"SUCCESS: Debug image saved to: {output_path}")
            print(f"BBox found: {meta.bbox}")
            print(f"Method: {meta.method}")
        except Exception as e:
            print(f"ERROR: {str(e)}")