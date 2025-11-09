# backend/app.py
import os
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from pathlib import Path
import sys, io, contextlib
import pandas as pd
import shutil

# ====== FastAPI Base ======
app = FastAPI()

# ====== CORS（把 127.0.0.1 也放进来）======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Directory preparation: final_result is exposed as a static download directory ======
BASE_DIR = Path(__file__).parent
MODULES_DIR = BASE_DIR / "modules"
FINAL_DIR = MODULES_DIR / "final_result"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = MODULES_DIR / "gee_out"
app.mount("/static", StaticFiles(directory=str(FINAL_DIR)), name="static")

# ====== Add modules to PYTHONPATH and import myMain ======
if str(MODULES_DIR) not in sys.path:
    sys.path.append(str(MODULES_DIR))
from modules.myMain import main as run_pipeline 

# ====== Request body in front-end coordinate mode ======
class S2Request(BaseModel):
    minLon: float
    minLat: float
    maxLon: float
    maxLat: float
    startDate: str   # "YYYY-MM-DD"
    endDate: str     # "YYYY-MM-DD"


# ====== Tool: Run a pipeline and return the result path ======
def _run_and_pick_latest_gpkg(argv_list):
    """
    Run the complete pipeline and return a dictionary containing the paths to three files：
    merged_gpkg, preview_png, field_summary_xlsx
    """
    sys.argv = ["myMain.py", *argv_list]

    # Run pipeline(now run_pipeline = myMain.main)
    result = run_pipeline() 

    # If the pipeline does not return a result, then the latest file is retrieved from final_result as a fallback.
    if not result or not isinstance(result, dict):
        gpkg_list = sorted(FINAL_DIR.glob("merged_*.gpkg"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        png_list = sorted(FINAL_DIR.glob("preview*.png"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        xlsx_list = sorted(FINAL_DIR.glob("field_summary_*.xlsx"),
                            key=lambda p: p.stat().st_mtime, reverse=True)

        if not gpkg_list:
            raise RuntimeError("No merged_*.gpkg found under final_result.")
        
        if not png_list:
            raise RuntimeError("No preview*.png found under final_result.")
        
        if not xlsx_list:
            raise RuntimeError("No field_summary_*.xlsx found under final_result.")
        result = {
            "merged_gpkg": gpkg_list[0].name,
            "preview_png": png_list[0].name if png_list else None,
            "field_summary_xlsx": xlsx_list[0].name if xlsx_list else None,
        }
    return result

    # gpkg_list = sorted(FINAL_DIR.glob("merged_*.gpkg"),
    #                        key=lambda p: p.stat().st_mtime, reverse=True)
    # png_list = sorted(FINAL_DIR.glob("preview*.png"),
    #                         key=lambda p: p.stat().st_mtime, reverse=True)
    # xlsx_list = sorted(FINAL_DIR.glob("field_summary_*.xlsx"),
    #                         key=lambda p: p.stat().st_mtime, reverse=True)

    # if not gpkg_list:
    #         raise RuntimeError("No merged_*.gpkg found under final_result.")
        
    # if not png_list:
    #         raise RuntimeError("No preview*.png found under final_result.")
        
    # if not xlsx_list:
    #         raise RuntimeError("No field_summary_*.xlsx found under final_result.")
    # result = {
    #         "merged_gpkg": gpkg_list[0].name,
    #         "preview_png": png_list[0].name if png_list else None,
    #         "field_summary_xlsx": xlsx_list[0].name if xlsx_list else None,
    #     }
    
    # return result

# =========================
# 1) coordinate mode(JSON)
# =========================
@app.post("/api/s2-process")
async def s2_process(req: S2Request):
    try:
        argv = [
            "--mode", "auto",              
            "--roi_mode", "bbox",          
            "--gee_bbox", f"{req.minLon},{req.minLat},{req.maxLon},{req.maxLat}",
            "--gee_s2_start", req.startDate,
            "--gee_s2_end", req.endDate,
        ]

        # Run the whole pipeline
        result = _run_and_pick_latest_gpkg(argv)

        # Retrieve filenames from the returned result dictionary.
        gpkg_name = os.path.basename(result["merged_gpkg"])
        png_name = os.path.basename(result["preview_png"])
        xlsx_name = os.path.basename(result["field_summary_xlsx"])
        

        # Construct a static URL
        base_url = "http://127.0.0.1:5000/static"
        gpkg_url = f"{base_url}/{gpkg_name}"
        preview_url = f"{base_url}/{png_name}"
        xlsx_url = f"{base_url}/{xlsx_name}"

        xlsx_path = FINAL_DIR / xlsx_name

        df = pd.read_excel(xlsx_path)
        top5 = df.head(5).to_dict(orient="records")

        # Return to frontend
        return {
            "status": "done",
            "gpkgUrl": gpkg_url,
            "gpkgName": gpkg_name,
            "previewUrl": preview_url,
            "xlsxUrl": xlsx_url,
            "xlsxName": xlsx_name,
            "top5": top5
        }

    except Exception as e:
        print(f"[ERROR] s2_process failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/s2-upload")
async def s2_upload(
    file: UploadFile = File(...),
    startDate: str = Form(...),
    endDate: str = Form(...),
):
    try:
        upload_dir = Path(OUT_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        # --- New name: S2_RGB8 + original suffix (default is .tif) ---
        ext = Path(file.filename).suffix or ".tif"
        new_filename = f"S2_RGB8{ext}"
        tif_path = upload_dir / new_filename

        try:
            file.file.seek(0) 
        except Exception:
            pass
        with open(tif_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"[SAVE] wrote tif to: {tif_path} | exists={tif_path.exists()}")
        tif_abs = tif_path.resolve()
        modules_dir = Path(__file__).resolve().parent / "modules"
        relative_tif = tif_abs.relative_to(modules_dir) if tif_abs.is_relative_to(modules_dir) else tif_abs
        print(f"[UPLOAD] Received file: {tif_path}")

        rel_for_pipeline = (
            ("modules" / relative_tif) if isinstance(relative_tif, Path) else Path("modules") / relative_tif
        ).as_posix() if not str(relative_tif).startswith(("modules/", "modules\\")) else Path(str(relative_tif)).as_posix()

        argv = [
            "--mode", "auto",
            "--roi_mode", "tif",
            "--user_tif", f"./{rel_for_pipeline}",
            "--gee_s2_start", startDate,
            "--gee_s2_end", endDate,
        ]

        print("--gee_s2_start:" + startDate)
        print("--gee_s2_end:" + endDate)

        print("--user_tif:" + f"./{rel_for_pipeline}")

        print(argv)

         # === Run the program in the backend directory, ensuring the relative path is correct ===
        _backend_dir = Path(__file__).resolve().parent
        _old_cwd = Path.cwd()
        try:
            os.chdir(_backend_dir)
            # use pipeline（myMain.main）
            result = _run_and_pick_latest_gpkg(argv)
        finally:
            os.chdir(_old_cwd)

        # Retrieve the filename of the result
        gpkg_name = result["merged_gpkg"]
        png_name = result["preview_png"]
        xlsx_name = result["field_summary_xlsx"]
        xlsx_path = FINAL_DIR / xlsx_name

        # Construct a URL
        base_url = "http://127.0.0.1:5000/static"
        gpkg_url = f"{base_url}/{gpkg_name}"
        preview_url = f"{base_url}/{png_name}"
        xlsx_url = f"{base_url}/{xlsx_name}"

        # Read the statistics of the first five lines
        df = pd.read_excel(xlsx_path)
        top5 = df.head(5).to_dict(orient="records")
        if tif_path.exists():
            tif_path.unlink()
            print(f"[CLEAN] Deleted uploaded file: {tif_path}")

        return {
            "status": "done",
            "previewUrl": preview_url,
            "gpkgUrl": gpkg_url,
            "gpkgName": gpkg_name,
            "xlsxUrl": xlsx_url,
            "xlsxName": xlsx_name,
            "top5": top5,
        }

    except Exception as e:
        print(f"[ERROR] s2_upload failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
