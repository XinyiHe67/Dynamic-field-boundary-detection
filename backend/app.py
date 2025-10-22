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

# ====== FastAPI 基础 ======
app = FastAPI()

# ====== CORS（把 127.0.0.1 也放进来）======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 目录准备：final_result 暴露为静态下载目录 ======
BASE_DIR = Path(__file__).parent
MODULES_DIR = BASE_DIR / "modules"
FINAL_DIR = MODULES_DIR / "final_result"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = MODULES_DIR / "gee_out"
app.mount("/static", StaticFiles(directory=str(FINAL_DIR)), name="static")

# ====== 将 modules 加到 PYTHONPATH 并导入 myMain ======
if str(MODULES_DIR) not in sys.path:
    sys.path.append(str(MODULES_DIR))
from modules.myMain import main as run_pipeline  # 你复制过来的 myMain.main()

# ====== 前端坐标模式的请求体（你之前写了 S2Request 但没定义）======
class S2Request(BaseModel):
    minLon: float
    minLat: float
    maxLon: float
    maxLat: float
    startDate: str   # "YYYY-MM-DD"
    endDate: str     # "YYYY-MM-DD"


# ====== 工具：跑 pipeline 并返回结果路径 ======
def _run_and_pick_latest_gpkg(argv_list):
    """
    运行完整 pipeline，并返回包含 3 个文件路径的字典：
    merged_gpkg, preview_png, field_summary_xlsx
    """
    sys.argv = ["myMain.py", *argv_list]

    # 运行 pipeline（此时 run_pipeline = myMain.main）
    result = run_pipeline()  # 🚀 它会返回一个字典

    # 若 pipeline 没返回结果，则兜底从 final_result 中取最新文件
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
# 1) 坐标模式（JSON）
# =========================
@app.post("/api/s2-process")
async def s2_process(req: S2Request):
    """
    前端传坐标 + 时间（JSON），运行 myMain，
    并返回生成的 gpkg 与 png 的静态访问 URL。
    """
    try:
        argv = [
            "--mode", "auto",              # 一键流程
            "--roi_mode", "bbox",          # 坐标模式
            "--gee_bbox", f"{req.minLon},{req.minLat},{req.maxLon},{req.maxLat}",
            "--gee_s2_start", req.startDate,
            "--gee_s2_end", req.endDate,
        ]

        # 执行完整 pipeline
        result = _run_and_pick_latest_gpkg(argv)

        # 从返回的结果字典中取文件名
        gpkg_name = os.path.basename(result["merged_gpkg"])
        png_name = os.path.basename(result["preview_png"])
        xlsx_name = os.path.basename(result["field_summary_xlsx"])
        

        # 拼成静态 URL
        base_url = "http://127.0.0.1:5000/static"
        gpkg_url = f"{base_url}/{gpkg_name}"
        preview_url = f"{base_url}/{png_name}"
        xlsx_url = f"{base_url}/{xlsx_name}"

        xlsx_path = FINAL_DIR / xlsx_name

        df = pd.read_excel(xlsx_path)
        top5 = df.head(5).to_dict(orient="records")

        # 返回给前端
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
    """
    前端上传 GeoTIFF 文件 + 时间范围，后端运行 myMain 的 tif 模式。
    """
    try:
        # 保存上传文件到临时路径
        upload_dir = Path(OUT_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        # --- 新名字：S2_RGB8 + 原后缀（默认为 .tif） ---
        ext = Path(file.filename).suffix or ".tif"
        new_filename = f"S2_RGB8{ext}"
        tif_path = upload_dir / new_filename
        # === 真的把上传的内容写入磁盘（Windows/mac 都一样）===

        try:
            file.file.seek(0)  # 保险：光标复位
        except Exception:
            pass
        with open(tif_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"[SAVE] wrote tif to: {tif_path} | exists={tif_path.exists()}")
        # ===============================================
        # 目标是：tif_path 相对于 backend/modules/
        tif_abs = tif_path.resolve()
        modules_dir = Path(__file__).resolve().parent / "modules"
        relative_tif = tif_abs.relative_to(modules_dir) if tif_abs.is_relative_to(modules_dir) else tif_abs
        print(f"[UPLOAD] Received file: {tif_path}")

        # === 新增：构造对管线稳定的相对路径（相对 backend 根），并统一斜杠 ===
        # 如果 relative_tif 形如 "final_result/xxx.tif"，前面补上 "modules/"
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

         # === 新增：在 backend 目录下运行，保证相对路径正确 ===
        _backend_dir = Path(__file__).resolve().parent
        _old_cwd = Path.cwd()
        try:
            os.chdir(_backend_dir)
            # 调用 pipeline（myMain.main）
            result = _run_and_pick_latest_gpkg(argv)
        finally:
            os.chdir(_old_cwd)

        # 拿出结果文件名
        gpkg_name = result["merged_gpkg"]
        png_name = result["preview_png"]
        xlsx_name = result["field_summary_xlsx"]
        xlsx_path = FINAL_DIR / xlsx_name

        # 拼成 URL
        base_url = "http://127.0.0.1:5000/static"
        gpkg_url = f"{base_url}/{gpkg_name}"
        preview_url = f"{base_url}/{png_name}"
        xlsx_url = f"{base_url}/{xlsx_name}"

        # 读出前五行统计
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