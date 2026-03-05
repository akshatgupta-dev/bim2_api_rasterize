from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import numpy as np
import os

from chi.rasterize_structural import main as calculate_alignment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def apply_T(T: np.ndarray, x: float, y: float):
    p = T @ np.array([x, y, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


@app.post("/api/align")
async def align_files(pdf: UploadFile = File(...), ifc: UploadFile = File(...)):
    pdf_path = f"temp_{pdf.filename}"
    ifc_path = f"temp_{ifc.filename}"

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(pdf.file, buffer)

    with open(ifc_path, "wb") as buffer:
        shutil.copyfileobj(ifc.file, buffer)

    try:
        result = calculate_alignment(pdf_path, ifc_path)

        # IMPORTANT:
        # This must be the fixed transform:
        # PDF PAGE POINTS (Y-up) -> IFC WORLD coordinates
        T_pdfPage_to_ifcWorld = np.array(result["T_pdfPage_to_ifcWorld"], dtype=np.float64)

        pdf_meta = result["pdf_meta"]
        ifc_meta = result["ifc_meta"]

        pdf_w = float(pdf_meta.page_width)
        pdf_h = float(pdf_meta.page_height)

        scale_to_m = float(getattr(ifc_meta, "unit_scale_to_m", 1.0))

        def map_pdf_page_pt_to_ifc_world(px, py):
            """
            Input:
              px, py = PDF page coordinates in POINTS, Y-up
            Output:
              IFC world coordinates (converted to meters if IFC units are mm)
            """
            x_ifc, y_ifc = apply_T(T_pdfPage_to_ifcWorld, px, py)
            return float(x_ifc ), float(y_ifc)

        # PDF page corners in Y-up page coordinates
        bl_x, bl_z = map_pdf_page_pt_to_ifc_world(0.0, 0.0)
        br_x, br_z = map_pdf_page_pt_to_ifc_world(pdf_w, 0.0)
        tr_x, tr_z = map_pdf_page_pt_to_ifc_world(pdf_w, pdf_h)
        tl_x, tl_z = map_pdf_page_pt_to_ifc_world(0.0, pdf_h)
        print("unit_scale_to_m:", scale_to_m)
        print("corners (raw units):", {
        "tl": apply_T(T_pdfPage_to_ifcWorld, 0, pdf_h),
        "tr": apply_T(T_pdfPage_to_ifcWorld, pdf_w, pdf_h),
        "br": apply_T(T_pdfPage_to_ifcWorld, pdf_w, 0),
        "bl": apply_T(T_pdfPage_to_ifcWorld, 0, 0),
        })
        return {
            "success": True,
            "pdf_coord_system": {
                "units": "pt",
                "y_axis": "up",
                "page_width": pdf_w,
                "page_height": pdf_h,
            },
            "corners": {
                "tl": [tl_x, tl_z],
                "tr": [tr_x, tr_z],
                "br": [br_x, br_z],
                "bl": [bl_x, bl_z],
            },
            # Optional: include matrix too for debugging
            "T_pdfPage_to_ifcWorld": T_pdfPage_to_ifcWorld.tolist(),
        }

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if os.path.exists(ifc_path):
            os.remove(ifc_path)
            
