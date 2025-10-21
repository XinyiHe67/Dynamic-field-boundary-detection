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
    # 模拟返回结果
    return {
        "status": "done",
        "previewUrl": "http://127.0.0.1:5000/static/example-result.png",
        "gpkgName": "Boundary_string_to_string.gpkg",
        "fieldCount": 12,
        "fieldArea": 45.7,
    }


@app.post("/api/s2-upload")
async def s2_upload(file: UploadFile = File(...), startDate: str = Form(...), endDate: str = Form(...)):
    """
    接收图片上传请求（FormData 格式）
    """
    print(f"收到文件: {file.filename}, 时间范围: {startDate} ~ {endDate}")
    return {
        "status": "done",
        "previewUrl": "http://127.0.0.1:5000/static/example-result.png",
        "gpkgName": "upload_result.gpkg",
        "fieldCount": 9,
        "fieldArea": 32.5,
    }