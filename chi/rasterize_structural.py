import cv2
import numpy as np
import json
from datetime import datetime

from chi.pdf_edges import extract_pdf_edges
from chi.ifc_edges_floor1 import extract_ifc_plan_edges, flip_ifc_segments


def segments_to_image(
    segments,
    bbox_w,
    bbox_h,
    out_size=2048,
    thickness=2,
    margin=10,
    return_matrix=False
):
    """
    Rasterize Y-up segments into an image. Produces matrix A such that:
      [px, py, 1]^T = A @ [x, y, 1]^T
    where (x,y) are in the segment coordinate system (Y-up),
    and (px,py) are image pixels (Y-down).
    """
    img = np.zeros((out_size, out_size), dtype=np.uint8)

    sx = (out_size - 2 * margin) / max(bbox_w, 1e-6)
    sy = (out_size - 2 * margin) / max(bbox_h, 1e-6)

    # px = margin + sx*x
    # py = margin + sy*(bbox_h - y) = margin + sy*bbox_h - sy*y
    A = np.array([
        [sx,  0,  margin],
        [0,  -sy, margin + sy * bbox_h],
        [0,   0,  1.0]
    ], dtype=np.float64)

    for x1, y1, x2, y2 in segments:
        p1 = A @ np.array([x1, y1, 1.0], dtype=np.float64)
        p2 = A @ np.array([x2, y2, 1.0], dtype=np.float64)
        cv2.line(
            img,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            255,
            thickness,
            lineType=cv2.LINE_AA
        )

    if return_matrix:
        return img, A
    return img


def ecc_align(moving, fixed, motion=cv2.MOTION_AFFINE, n_iter=3000):
    """
    Find warp that aligns 'moving' onto 'fixed' using ECC maximization.

    IMPORTANT detail:
    We apply warpAffine(..., flags=WARP_INVERSE_MAP), matching the common OpenCV ECC usage.
    In this setup, the returned warp behaves like a mapping from FIXED -> MOVING
    (destination -> source) in pixel space.

    Returns:
      warp (2x3), aligned_moving (same size as fixed), ecc_score
    """
    moving_f = moving.astype(np.float32) / 255.0
    fixed_f = fixed.astype(np.float32) / 255.0

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, 1e-7)

    cc, warp = cv2.findTransformECC(fixed_f, moving_f, warp, motion, criteria)

    h, w = fixed.shape[:2]
    aligned = cv2.warpAffine(
        moving,
        warp,
        (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )
    return warp, aligned, cc


def chamfer_ifc_to_pdf_trimmed(pdf_edges_u8, ifc_edges_u8, trim_q=90):
    """
    One-way IFC->PDF Chamfer, robust to PDF clutter and some unmatched IFC edges.
    Keeps the best trim_q% distances (drops worst 100-trim_q%).
    """
    pdf_bin = (pdf_edges_u8 > 0).astype(np.uint8)
    ifc_bin = (ifc_edges_u8 > 0).astype(np.uint8)

    dt = cv2.distanceTransform(1 - pdf_bin, cv2.DIST_L2, 3)
    ys, xs = np.where(ifc_bin > 0)
    if len(xs) == 0:
        return None

    d = dt[ys, xs]
    cutoff = np.percentile(d, trim_q)
    d_trim = d[d <= cutoff]

    return {
        "type": "ifc_to_pdf_trimmed_chamfer",
        "trim_q": float(trim_q),
        "mean_px": float(np.mean(d_trim)),
        "median_px": float(np.median(d_trim)),
        "p90_px": float(np.percentile(d_trim, 90)),
        "within_2px": float(np.mean(d_trim <= 2.0)),
        "within_5px": float(np.mean(d_trim <= 5.0)),
        "n_points": int(len(d)),
        "n_used": int(len(d_trim)),
        "cutoff_px": float(cutoff),
    }


def remove_small_components(bin_img_u8, min_area=50):
    """
    Removes small connected components (text specks, dots).
    Keeps only components with pixel area >= min_area.
    """
    bin01 = (bin_img_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)

    out = np.zeros_like(bin_img_u8)
    for i in range(1, num):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def warp2x3_to_3x3(w2x3):
    W = np.eye(3, dtype=np.float64)
    W[:2, :] = np.asarray(w2x3, dtype=np.float64)
    return W


def shift_matrix(tx, ty):
    """
    Homogeneous translation matrix:
      [x',y',1]^T = S @ [x,y,1]^T
    """
    return np.array([
        [1.0, 0.0, float(tx)],
        [0.0, 1.0, float(ty)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def save_alignment_report(
    *,
    pdf_path,
    ifc_path,
    pdf_meta,
    ifc_meta,
    out_size,
    margin,
    score_aff,
    warp_aff,
    score_euc,
    warp_euc,
    metrics,
    A_pdf,
    A_ifc,
    W_pdfPix_to_ifcPix,
    T_ifcLocal_to_pdfLocal,
    T_ifcWorld_to_pdfPage,
    output_file="alignment_results.json"
):
    """
    We save BOTH:
      - local<->local transforms (debug)
      - world<->page transforms (what your viewer should use)
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "pdf_path": pdf_path,
        "ifc_path": ifc_path,

        "config": {
            "out_size": int(out_size),
            "margin": int(margin),
        },

        # ECC results
        "ecc_affine_score": float(score_aff),
        "ecc_affine_warp": np.asarray(warp_aff, dtype=float).tolist(),
        "ecc_euclidean_score": float(score_euc),
        "ecc_euclidean_warp": np.asarray(warp_euc, dtype=float).tolist(),

        "metrics": metrics,

        # Rasterization matrices (segment coords -> raster px coords)
        "A_pdf": np.asarray(A_pdf, dtype=float).tolist(),
        "A_ifc": np.asarray(A_ifc, dtype=float).tolist(),

        # Combined pixel-space warp (PDF pixels -> IFC pixels)
        "W_pdfPix_to_ifcPix": np.asarray(W_pdfPix_to_ifcPix, dtype=float).tolist(),

        # Debug transform: IFC local -> PDF local
        "T_ifcLocal_to_pdfLocal": np.asarray(T_ifcLocal_to_pdfLocal, dtype=float).tolist(),

        # Viewer-facing transforms
        "T_ifcWorld_to_pdfPage": np.asarray(T_ifcWorld_to_pdfPage, dtype=float).tolist(),
        "T_pdfPage_to_ifcWorld": np.linalg.inv(T_ifcWorld_to_pdfPage).astype(float).tolist(),

        "pdf_meta": {
            "method": getattr(pdf_meta, "method", "unknown"),
            "page_width_pt": float(pdf_meta.page_width),
            "page_height_pt": float(pdf_meta.page_height),
            "bbox": [float(x) for x in pdf_meta.bbox],
            "shift": [float(pdf_meta.shift[0]), float(pdf_meta.shift[1])],
            "coord_system": {"units": "pt", "y_axis": "up"},
        },
        "ifc_meta": {
            "bbox": [float(x) for x in ifc_meta.bbox],
            "shift": [float(ifc_meta.shift[0]), float(ifc_meta.shift[1])],
            "unit_name": getattr(ifc_meta, "unit_name", "unknown"),
            "unit_scale_to_m": float(getattr(ifc_meta, "unit_scale_to_m", 1.0)),
        },
    }

    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except Exception:
        data = []

    data.append(report)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nSaved alignment report to {output_file}")


def main(pdf_path, ifc_path, out_size=2048, margin=10):
    # 1) Extract segments and metas
    pdf_segs, pdf_meta = extract_pdf_edges(pdf_path)
    ifc_segs, ifc_meta = extract_ifc_plan_edges(ifc_path)

    # Optional flip if needed
    # ifc_segs = flip_ifc_segments(ifc_segs, ifc_meta)

    # 2) Rasterize both to same canvas
    pdf_img, A_pdf = segments_to_image(
        pdf_segs, pdf_meta.bbox[2], pdf_meta.bbox[3],
        out_size=out_size, margin=margin, return_matrix=True
    )
    ifc_img, A_ifc = segments_to_image(
        ifc_segs, ifc_meta.bbox[2], ifc_meta.bbox[3],
        out_size=out_size, margin=margin, return_matrix=True
    )

    # 3) Clean PDF specks
    pdf_img = remove_small_components(pdf_img, min_area=60)

    # Save inputs for inspection
    cv2.imwrite("01_pdf_raster.png", pdf_img)
    cv2.imwrite("02_ifc_raster.png", ifc_img)

    # 4) Dilation helps ECC
    k = np.ones((3, 3), np.uint8)
    pdf_for = cv2.dilate(pdf_img, k, iterations=1)
    ifc_for = cv2.dilate(ifc_img, k, iterations=1)

    # 5) ECC affine alignment
    warp_aff, ifc_aff, score_aff = ecc_align(ifc_for, pdf_for, motion=cv2.MOTION_AFFINE)
    print("ECC (AFFINE) score:", score_aff)
    print("AFFINE warp:\n", warp_aff)

    # 6) ECC euclidean refinement
    warp_euc, ifc_rigid, score_euc = ecc_align(ifc_aff, pdf_for, motion=cv2.MOTION_EUCLIDEAN)
    print("\nECC (EUCLIDEAN) score:", score_euc)
    print("EUCLIDEAN warp:\n", warp_euc)

    # Combine warps correctly
    # Because of WARP_INVERSE_MAP, these act like PDF_pix -> IFC_pix
    W_aff = warp2x3_to_3x3(warp_aff)
    W_euc = warp2x3_to_3x3(warp_euc)
    W_pdfPix_to_ifcPix = W_aff @ W_euc

    # For coordinate transforms we need IFC_pix -> PDF_pix
    M_ifcPix_to_pdfPix = np.linalg.inv(W_pdfPix_to_ifcPix)

    # IFC local -> PDF local
    T_ifcLocal_to_pdfLocal = np.linalg.inv(A_pdf) @ M_ifcPix_to_pdfPix @ A_ifc

    # Wrap with shifts to get IFC world -> PDF page points (Y-up)
    S_ifcWorld_to_local = shift_matrix(-float(ifc_meta.shift[0]), -float(ifc_meta.shift[1]))
    S_pdfLocal_to_page = shift_matrix(float(pdf_meta.shift[0]), float(pdf_meta.shift[1]))

    T_ifcWorld_to_pdfPage = S_pdfLocal_to_page @ T_ifcLocal_to_pdfLocal @ S_ifcWorld_to_local
    T_pdfPage_to_ifcWorld = np.linalg.inv(T_ifcWorld_to_pdfPage)

    # 7) Quantitative check
    ifc_aligned = ifc_rigid
    metrics = chamfer_ifc_to_pdf_trimmed(pdf_img, ifc_aligned, trim_q=90)

    print("\nChamfer alignment metrics:")
    if metrics is None:
        print("No IFC pixels found after warp.")
    else:
        for k_, v_ in metrics.items():
            print(f"{k_}: {v_}")

    # 8) Save outputs
    cv2.imwrite("03_ifc_aligned.png", ifc_aligned)
    overlay = cv2.merge([pdf_img, ifc_aligned, np.zeros_like(pdf_img)])
    cv2.imwrite("04_overlay.png", overlay)

    # 9) Save JSON report
    save_alignment_report(
        pdf_path=pdf_path,
        ifc_path=ifc_path,
        pdf_meta=pdf_meta,
        ifc_meta=ifc_meta,
        out_size=out_size,
        margin=margin,
        score_aff=score_aff,
        warp_aff=warp_aff,
        score_euc=score_euc,
        warp_euc=warp_euc,
        metrics=metrics,
        A_pdf=A_pdf,
        A_ifc=A_ifc,
        W_pdfPix_to_ifcPix=W_pdfPix_to_ifcPix,
        T_ifcLocal_to_pdfLocal=T_ifcLocal_to_pdfLocal,
        T_ifcWorld_to_pdfPage=T_ifcWorld_to_pdfPage,
        output_file="alignment_results.json"
    )

    # Optional quality flag for caller
    quality_ok = False
    if metrics is not None:
        quality_ok = (
            metrics["mean_px"] < 6.0 and
            metrics["within_5px"] > 0.70
        )

    # RETURN VALUE USED BY API
    return {
        "success": True,
        "quality_ok": quality_ok,

        "pdf_path": pdf_path,
        "ifc_path": ifc_path,

        # Keep as Python objects so API can do pdf_meta.page_width etc.
        "pdf_meta": pdf_meta,
        "ifc_meta": ifc_meta,

        "config": {
            "out_size": out_size,
            "margin": margin,
        },

        "score_aff": float(score_aff),
        "score_euc": float(score_euc),
        "metrics": metrics,

        # Arrays / matrices
        "warp_aff": warp_aff,
        "warp_euc": warp_euc,
        "A_pdf": A_pdf,
        "A_ifc": A_ifc,
        "W_pdfPix_to_ifcPix": W_pdfPix_to_ifcPix,
        "T_ifcLocal_to_pdfLocal": T_ifcLocal_to_pdfLocal,

        # Main matrices you want downstream
        "T_ifcWorld_to_pdfPage": T_ifcWorld_to_pdfPage,
        "T_pdfPage_to_ifcWorld": T_pdfPage_to_ifcWorld,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 rasterize.py drawing.pdf model.ifc")
        sys.exit(1)

    result = main(sys.argv[1], sys.argv[2])
    print("\nReturned keys:", list(result.keys()))