from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional
import numpy as np
import ifcopenshell
import ifcopenshell.geom
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # prevents Qt/xcb crash
import matplotlib.pyplot as plt

Segment = Tuple[float, float, float, float]
SegmentZ = Tuple[float, float, float, float, float]  # (x1,y1,x2,y2,z)

@dataclass
class IfcEdgesMeta:
    unit_name: str
    unit_scale_to_m: float
    bbox: Tuple[float, float, float, float]  # (minx, miny, w, h) AFTER shift
    shift: Tuple[float, float]               # (shift_x, shift_y) applied to original coords
    element_counts: Dict[str, int]

def _get_ifc_units(ifc) -> Tuple[str, float]:
    try:
        projects = ifc.by_type("IfcProject")
        if not projects:
            return ("unknown", 1.0)
        uic = projects[0].UnitsInContext
        if not uic:
            return ("unknown", 1.0)
        for u in uic.Units:
            if u.is_a("IfcSIUnit") and getattr(u, "UnitType", None) == "LENGTHUNIT":
                prefix = getattr(u, "Prefix", None)
                name = getattr(u, "Name", None)
                if name and "METRE" in str(name).upper():
                    if prefix and str(prefix).upper() == "MILLI":
                        return ("mm", 0.001)
                    return ("m", 1.0)
        return ("unknown", 1.0)
    except Exception:
        return ("unknown", 1.0)

def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n / norm

def _mesh_feature_edges_xy(verts: np.ndarray, faces: np.ndarray, angle_deg: float = 25.0):
    """
    Keep edges that are either:
      - boundary edges (only 1 adjacent face), OR
      - 'sharp' edges where adjacent face normals differ by > angle_deg
    This removes most triangulation diagonals on flat surfaces.
    """
    normals = _face_normals(verts, faces)

    edge_to_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        tri = [(int(a), int(b)), (int(b), int(c)), (int(c), int(a))]
        for u, v in tri:
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            edge_to_faces[key].append(fi)

    cos_thr = np.cos(np.deg2rad(angle_deg))
    keep = set()

    for (i, j), fis in edge_to_faces.items():
        if len(fis) == 1:
            keep.add((i, j))  # boundary
        elif len(fis) >= 2:
            f1, f2 = fis[0], fis[1]
            c = float(np.dot(normals[f1], normals[f2]))
            if c < cos_thr:
                keep.add((i, j))

    return keep

def extract_ifc_plan_edges(
    ifc_path: str,
    include_types: Optional[Iterable[str]] = None,
    max_elements: Optional[int] = None,
    split_z_ref: float = 127.1,   # your reference split height
) -> Tuple[List[Segment], IfcEdgesMeta]:
    """
    Extract plan edges for FLOOR 1 ONLY using Z split:
      split_z = 127.1 if z_min < 127.1 < z_max else (z_threshold + z_max)/2
      ranges = [(z_threshold, split_z), (split_z, z_max+1)]
    Keeps only segments with avg_z in ranges[0].
    """

    if include_types is None:
        include_types = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcBuildingElementProxy"]

    ifc = ifcopenshell.open(ifc_path)
    unit_name, unit_scale_to_m = _get_ifc_units(ifc)

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    raw_segments: List[SegmentZ] = []
    z_values: List[float] = []
    element_counts: Dict[str, int] = defaultdict(int)

    total = 0
    for t in include_types:
        for el in ifc.by_type(t):
            if max_elements is not None and total >= max_elements:
                break
            total += 1

            try:
                shape = ifcopenshell.geom.create_shape(settings, el)
                verts = np.asarray(shape.geometry.verts).reshape((-1, 3))
                faces = np.asarray(shape.geometry.faces).reshape((-1, 3))

                avg_z = float(np.mean(verts[:, 2]))
                edges_idx = _mesh_feature_edges_xy(verts, faces, angle_deg=25.0)

                for i, j in edges_idx:
                    raw_segments.append((
                        float(verts[i, 0]), float(verts[i, 1]),
                        float(verts[j, 0]), float(verts[j, 1]),
                        avg_z
                    ))
                z_values.append(avg_z)
                element_counts[t] += 1
            except Exception:
                continue

    if not raw_segments or not z_values:
        raise RuntimeError("No segments extracted.")

    # --- Z SPLIT LOGIC (your exact rule) ---
    z_min = float(min(z_values))
    z_max = float(max(z_values))
    z_range = z_max - z_min

    # Ignore bottom 2% (site/ground junk)
    z_threshold = z_min + (z_range * 0.02)

    split_z = split_z_ref if (z_min < split_z_ref < z_max) else (z_threshold + z_max) / 2.0

    ranges = [
        (z_threshold, split_z),     # Floor 1
        (split_z, z_max + 1.0)      # Floor 2
    ]

    floor1_min, floor1_max = ranges[0]

    # --- FLOOR 1 FILTER ---
    floor1_segments: List[Segment] = []
    for x1, y1, x2, y2, z in raw_segments:
        if floor1_min <= z < floor1_max:
            floor1_segments.append((x1, y1, x2, y2))

    # Fallbacks (in case avg_z is weird for your model)
    if not floor1_segments:
        # fallback 1: just remove site/ground
        floor1_segments = [(x1, y1, x2, y2) for (x1, y1, x2, y2, z) in raw_segments if z > z_threshold]

    if not floor1_segments:
        # fallback 2: give up and keep all
        floor1_segments = [(x1, y1, x2, y2) for (x1, y1, x2, y2, z) in raw_segments]

    # --- RE-CENTER AND BBOX ---
    pts = np.array(
        [[s[0], s[1]] for s in floor1_segments] + [[s[2], s[3]] for s in floor1_segments],
        dtype=float
    )
    minx, miny = np.min(pts, axis=0)
    maxx, maxy = np.max(pts, axis=0)

    shifted_segments = [(x1 - minx, y1 - miny, x2 - minx, y2 - miny) for (x1, y1, x2, y2) in floor1_segments]

    meta = IfcEdgesMeta(
        unit_name=unit_name,
        unit_scale_to_m=unit_scale_to_m,
        bbox=(0.0, 0.0, float(maxx - minx), float(maxy - miny)),
        shift=(float(minx), float(miny)),
        element_counts=dict(element_counts),
    )
    return shifted_segments, meta

def flip_ifc_segments(segments: List[Segment], ifc_meta: IfcEdgesMeta) -> List[Segment]:
    max_y = ifc_meta.bbox[3]
    return [(x1, max_y - y1, x2, max_y - y2) for (x1, y1, x2, y2) in segments]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ifc_edges.py <path_to_ifc>")
        raise SystemExit(1)

    path = sys.argv[1]
    print(f"Processing Floor 1 only: {path}")

    segs, meta = extract_ifc_plan_edges_floor1_only(path)

    # Debug plot
    plt.figure(figsize=(10, 10))
    for x1, y1, x2, y2 in segs:
        plt.plot([x1, x2], [y1, y2], linewidth=0.5)
    plt.title(f"IFC Floor 1 Only: {meta.bbox[2]:.2f}x{meta.bbox[3]:.2f} units")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.6)

    output_path = "/home/chidepnek/RoboAI/BIM/BIM2/backend/src/reg_3_opencv/working1floor/ifc_floor1_only.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"SUCCESS: Debug image saved to: {output_path}")
    print(f"BBox: {meta.bbox}")
    print(f"Shift applied: {meta.shift}")
    print(f"Counts: {meta.element_counts}")