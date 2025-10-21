from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from fastapi.staticfiles import StaticFiles

import uuid
import time

app = FastAPI()

# === 允许跨域 ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# === 定义坐标模式的数据模型 ===
class S2Request(BaseModel):
    minLon: float
    minLat: float
    maxLon: float
    maxLat: float
    startDate: str
    endDate: str


@app.post("/api/s2-process")
async def s2_process(req: S2Request):
    """
    接收坐标 + 日期请求（JSON 格式）
    """
    print(f"接收到请求: {req.dict()}")
    job_id = str(uuid.uuid4())[:8]
    # 模拟返回结果
    return {
        "status": "done",
        "previewUrl": "http://127.0.0.1:5000/static/example-result.png",
        "gpkgName": "Boundary_string_to_string.gpkg",
        "jobId": job_id,
        "fieldCount": 12,
        "fieldArea": 45.7,
    }


@app.post("/api/s2-upload")
async def s2_upload(file: UploadFile = File(...), startDate: str = Form(...), endDate: str = Form(...)):
    """
    接收图片上传请求（FormData 格式）
    """
    print(f"收到文件: {file.filename}, 时间范围: {startDate} ~ {endDate}")
    job_id = str(uuid.uuid4())[:8]
    return {
        "status": "done",
        "previewUrl": "http://127.0.0.1:5000/static/example-result.png",
        "gpkgName": "upload_result.gpkg",
        "jobId": job_id,
        "fieldCount": 9,
        "fieldArea": 32.5,
    }

# 模拟任务状态查询接口（前端可以轮询用）
@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    """
    查询任务状态接口
    """
    print(f"收到查询任务状态请求: {job_id}")
    # 模拟一个任务已完成
    return {"jobId": job_id, "status": "done", "progress": 100}


# 下载结果文件接口
@app.get("/api/job/{job_id}/download")
def download_job(job_id: str):
    """
    下载指定任务的处理结果 (.gpkg)
    """
    file_path = Path("backend/downloads/upload_result.gpkg")  # 你的文件路径

    # 检查文件是否存在
    if not file_path.exists():
        return {"error": f"Result file not found for job {job_id}"}

    print(f"正在下载任务 {job_id} 的结果文件: {file_path}")

    # 返回文件
    return FileResponse(
        path=file_path,
        filename=f"{job_id}_result.gpkg",       # 下载时的文件名
        media_type="application/octet-stream"   # 二进制流类型
    )