import { useState } from 'react';

/* ===================================================
   后端接口配置
   =================================================== */
const BASE_URL = 'http://127.0.0.1:5000';
const ENDPOINTS = {
  processArea: `${BASE_URL}/api/process-area`,
  processImage: `${BASE_URL}/api/process-image`,
  previewImage: `${BASE_URL}/api/preview-image`,
  job: id => `${BASE_URL}/api/job/${id}`,
  jobDownload: id => `${BASE_URL}/api/job/${id}/download`,
};

/* ============================
   全局宽度修复
   ============================ */
function GlobalWidthFix() {
  return (
    <style>{`
      html, body, #root {
        width: 100%;
        max-width: 100%;
        overflow-x: hidden;
        margin: 0;
        padding: 0;
      }
    `}</style>
  );
}

/* ============================
   首页
   ============================ */
function HomePage() {
  return (
    <div
      className="min-h-screen w-full flex flex-col overflow-hidden"
      style={{
        background:
          'linear-gradient(135deg,#EEF7FF 0%,#F5FAFF 50%,#F7F3FF 100%)',
      }}
    >
      <header className="w-full sticky top-0 z-10 bg-white/70 backdrop-blur border-b border-slate-200">
        <div className="px-[clamp(12px,2vw,24px)] py-3 flex items-center justify-between">
          <div className="flex items-center gap-2 text-lg font-semibold text-slate-800">
            <div className="rounded-full border border-sky-400 px-2 py-1 text-sky-600 font-bold">
              FB
            </div>
            Farmland Boundary Detection System
          </div>
          <nav className="space-x-6 text-sm font-medium text-sky-700">
            <a href="#/" className="hover:text-slate-900">
              Home
            </a>
            <a href="#/use" className="hover:text-emerald-600">
              Use
            </a>
          </nav>
        </div>
      </header>

      <main className="flex-1 w-full px-[clamp(12px,2vw,24px)] py-10 flex items-center justify-center">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-center w-full">
          <div>
            <h1 className="text-[clamp(30px,4vw,52px)] font-extrabold leading-tight text-slate-900">
              Detect farmland boundaries from imagery
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-cyan-500 to-emerald-500">
                fast, accurate, automated.
              </span>
            </h1>
            <p className="mt-4 text-slate-600 text-sm leading-relaxed">
              Upload imagery or specify a region of interest. The pipeline runs
              segmentation, postprocessing, and visualization — giving you clean
              farmland boundaries.
            </p>
            <div className="mt-6 flex gap-3 flex-wrap">
              <a
                href="#/use"
                className="rounded-full bg-gradient-to-r from-blue-600 to-emerald-500
                           !text-white visited:!text-white hover:!text-white focus:!text-white active:!text-white
                           no-underline px-6 py-2 text-sm font-medium shadow hover:brightness-110"
                style={{ color: '#fff' }}
              >
                Try it now
              </a>
            </div>
          </div>

          <div className="rounded-2xl bg-white/85 border border-slate-200 aspect-[16/9] flex items-center justify-center shadow-inner overflow-hidden">
            <img
              src="/hero.png"
              alt="Farmland boundary preview"
              className="w-full h-full object-cover"
            />
          </div>
        </div>
      </main>

      <footer className="w-full px-[clamp(12px,2vw,24px)] py-6 text-xs text-slate-500 border-t border-slate-200">
        © 2025 Farmland Boundary Detection System
      </footer>
    </div>
  );
}

/* ============================
   Use Page 
   ============================ */
function UsePage() {
  const [tab, setTab] = useState('coords');
  const [minLon, setMinLon] = useState('146.15');
  const [minLat, setMinLat] = useState('-34.25');
  const [maxLon, setMaxLon] = useState('146.55');
  const [maxLat, setMaxLat] = useState('-33.95');

  const [startDate, setStartDate] = useState('2024-06-01');
  const [endDate, setEndDate] = useState('2024-09-01');

  const [coordError, setCoordError] = useState(''); // 统一错误提示

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState(null);

  const [resultUrl, setResultUrl] = useState(null);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);

  const [fieldCount, setFieldCount] = useState(null);
  const [fieldArea, setFieldArea] = useState(null);

  // —— 下载所需
  const [gpkgUrl, setGpkgUrl] = useState(null); // 后端返回的 .gpkg 直链
  const [gpkgName, setGpkgName] = useState('result.gpkg');

  // 使用页卡片布局
  const leftCardMinH =
    tab === 'coords' ? 'min(60vh, 640px)' : 'min(49vh, 500px)';
  const rightCardMinH =
    tab === 'coords' ? 'min(56vh, 600px)' : 'min(45vh, 480px)';
  const uploadPreviewH =
    tab === 'coords' ? 'min(34vh, 340px)' : 'min(27vh, 280px)';
  const resultPreviewH =
    tab === 'coords' ? 'min(34vh, 340px)' : 'min(27vh, 280px)';

  // 可下载条件
  const canDownload = status === 'done' && gpkgUrl;

  function handlePickFile(file) {
    if (!file.type.startsWith('image/')) {
      setCoordError('Only image files are supported.');
      return;
    }
    if (uploadPreviewUrl && uploadPreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(uploadPreviewUrl);
    }
    setSelectedFile(file || null);
    setUploadPreviewUrl(
      file?.type?.startsWith('image/') ? URL.createObjectURL(file) : null
    );
  }

  function validateCoords() {
    const minLonNum = parseFloat(minLon);
    const maxLonNum = parseFloat(maxLon);
    const minLatNum = parseFloat(minLat);
    const maxLatNum = parseFloat(maxLat);

    if (
      [minLon, maxLon, minLat, maxLat].some(
        v => v.trim() === '' || isNaN(parseFloat(v))
      )
    ) {
      return 'Please enter valid numeric coordinates.';
    }

    if (minLonNum < -180 || maxLonNum > 180) {
      return 'Longitude must be between -180 and 180.';
    }
    if (minLatNum < -90 || maxLatNum > 90) {
      return 'Latitude must be between -90 and 90.';
    }

    if (minLonNum >= maxLonNum) {
      return 'Min Longitude must be smaller than Max Longitude.';
    }
    if (minLatNum >= maxLatNum) {
      return 'Min Latitude must be smaller than Max Latitude.';
    }
    return '';
  }

  /**
   * ===================================================
   * handleSubmit() — 触发边界检测任务
   * ===================================================
   * 前端提交用户输入（坐标或图片 + 时间范围），
   * 向后端发起请求，后端执行边界检测并返回结果。
   *
   * =============================
   * Request (前端 → 后端)
   * =============================
   * - 根据 tab 不同，调用不同的接口：
   *
   * 1. By Coordinates 模式:
   *    POST /api/s2-process
   *    Content-Type: application/json
   *    Body:
   *    {
   *      "minLon": 146.15,     // 最小经度
   *      "minLat": -34.25,     // 最小纬度
   *      "maxLon": 146.55,     // 最大经度
   *      "maxLat": -33.95,     // 最大纬度
   *      "startDate": "2024-06-01", // 起始时间 (YYYY-MM-DD)
   *      "endDate": "2024-09-01"    // 结束时间 (YYYY-MM-DD)
   *    }
   *
   * 2. By Image 模式:
   *    POST /api/s2-upload
   *    Content-Type: multipart/form-data
   *    Body:
   *    - "file": 上传的影像文件 (.tif / .png / .jpg)
   *    - "startDate": 起始日期
   *    - "endDate": 结束日期
   *
   * =============================
   * Response (后端 → 前端)
   * =============================
   * 成功 (HTTP 200):
   * {
   *   "previewUrl": "http://localhost:5000/outputs/example.png", // 预览图片 URL
   *   "gpkgUrl": "http://localhost:5000/downloads/result_001.gpkg", // 可下载的 GPKG 链接
   *   "gpkgName": "result_001.gpkg",   // 可选，文件名
   *   "fieldCount": 12,                // 检测出的农田数量
   *   "fieldArea": 45.8                // 总面积 (ha)
   * }
   *
   * 失败 (HTTP 4xx / 5xx):
   * {
   *   "error": "Invalid input" 或 "Internal Server Error"
   * }
   *
   * =============================
   * 后端接口判断逻辑建议
   * =============================
   * - 后端可以根据请求的 Content-Type 自动判断：
   *   - 若 Content-Type = "application/json" → 处理坐标请求；
   *   - 若 Content-Type = "multipart/form-data" → 处理图像上传请求。
   *
   *   例如在 Flask 或 FastAPI 中：
   *   ```
   *   if request.content_type.startswith("application/json"):
   *       data = request.json
   *       # 处理坐标模式
   *   elif request.content_type.startswith("multipart/form-data"):
   *       file = request.files["file"]
   *       # 处理影像模式
   *   ```
   *
   * =============================
   * 前端执行流程
   * =============================
   * 1. 验证输入合法性（坐标、时间范围）
   * 2. 设置加载状态与进度条
   * 3. 根据模式发送请求 (JSON / FormData)
   * 4. 等待后端返回结果，更新预览、统计信息、下载链接
   * 5. 若失败，显示错误提示
   */
  async function handleSubmit() {
    setCoordError('');

    if (tab === 'coords') {
      const errorMsg = validateCoords();
      if (errorMsg) {
        setCoordError(errorMsg);
        return;
      }
    } else {
      if (!selectedFile) {
        setCoordError('Please select an image file before submitting.');
        return;
      }
    }

    if (!startDate || !endDate) {
      setCoordError('Please select both start date and end date.');
      return;
    }

    const start = new Date(startDate);
    const end = new Date(endDate);
    if (isNaN(start) || isNaN(end)) {
      setCoordError('Invalid date format.');
      return;
    }

    if (start >= end) {
      setCoordError('Start date must be earlier than end date.');
      return;
    }

    setStatus('running');
    setProgress(0);
    setResultUrl(null);
    setGpkgUrl(null);

    // 启动“假进度条动画”，让用户看到加载过程
    const t = setInterval(() => {
      setProgress(p => Math.min(p + Math.random() * 10, 95)); // 先最多到95%
    }, 400);

    try {
      // === 调后端接口（预留位置） ===
      let response;

      if (tab === 'coords') {
        const body = {
          minLon,
          minLat,
          maxLon,
          maxLat,
          startDate,
          endDate,
        };
        response = await fetch(`${BASE_URL}/api/s2-process`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
      } else {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('startDate', startDate);
        formData.append('endDate', endDate);
        response = await fetch(`${BASE_URL}/api/s2-upload`, {
          method: 'POST',
          body: formData,
        });
      }

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      // === 成功后，结束动画并设置结果 ===
      clearInterval(t);
      setProgress(100);
      setStatus('done');
      setResultUrl(data.previewUrl || '/example-result.png');
      if (data.gpkgUrl) setGpkgUrl(data.gpkgUrl);
      if (data.gpkgName) setGpkgName(data.gpkgName);
      if (data.fieldCount) setFieldCount(data.fieldCount || 0);
      if (data.fieldArea) setFieldArea(data.fieldArea || 0);
    } catch (err) {
      console.error(err);
      clearInterval(t);
      setCoordError('Failed to connect to backend.');
      setStatus('idle');
      setProgress(0);
      setResultUrl(null);
      setFieldCount(null);
      setFieldArea(null);
    }

    // const t = setInterval(() => {
    //   setProgress(p => {
    //     const next = p + Math.random() * 12;
    //     if (next >= 100) {
    //       clearInterval(t);
    //       setProgress(100);
    //       setStatus('done');
    //       setResultUrl('/example-result.png');
    //       // TODO: 后端就绪后任选一种方式提供下载来源：
    //       // setGpkgUrl("https://YOUR_SERVER/path/to/result.gpkg"); // 方式1：直链
    //     }
    //     return Math.min(next, 100);
    //   });
    // }, 420);
  }

  // —— 下载逻辑
  async function handleDownload() {
    if (!(status === 'done')) return;

    // 方式1：已有直链 => 直接下载
    if (gpkgUrl) {
      const a = document.createElement('a');
      a.href = gpkgUrl;
      a.download = gpkgName || 'result.gpkg';
      document.body.appendChild(a);
      a.click();
      a.remove();
      return;
    }
  }

  return (
    <div
      className="min-h-screen w-full flex flex-col overflow-hidden"
      style={{
        background:
          'linear-gradient(135deg,#EEF7FF 0%,#F5FAFF 50%,#F7F3FF 100%)',
      }}
    >
      <header className="w-full sticky top-0 z-10 bg-white/70 backdrop-blur border-b border-slate-200">
        <div className="px-[clamp(12px,2vw,24px)] py-3 flex items-center justify-between">
          <div className="flex items-center gap-2 text-lg font-semibold text-slate-800">
            <div className="rounded-full border border-sky-400 px-2 py-1 text-sky-600 font-bold">
              FB
            </div>
            Farmland Boundary Detection System
          </div>
          <nav className="space-x-6 text-sm font-medium text-sky-700">
            <a href="#/" className="hover:text-slate-900">
              Home
            </a>
            <a href="#/use" className="hover:text-emerald-600">
              Use
            </a>
          </nav>
        </div>
      </header>

      <main className="flex-1 w-full px-[clamp(12px,2vw,24px)] py-10">
        <h1 className="text-[clamp(28px,3vw,44px)] font-semibold text-slate-900">
          Run the boundary-detection pipeline
        </h1>
        <p className="mt-2 text-slate-700 text-sm">
          Input coordinates or upload imagery. The backend will process and show
          a preview.
        </p>

        <div className="mt-8 grid grid-cols-1 xl:grid-cols-2 gap-6 w-full items-stretch">
          {/* 左卡片 */}
          <div
            className="w-full rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-md flex flex-col"
            style={{ minHeight: leftCardMinH }}
          >
            <div className="flex gap-3 mb-4">
              <button
                onClick={() => setTab('coords')}
                className={`rounded-xl px-4 py-1.5 text-sm font-medium ${
                  tab === 'coords'
                    ? 'bg-gradient-to-r from-blue-600 to-emerald-500 text-white'
                    : 'bg-white text-slate-700 border border-slate-300'
                }`}
              >
                By Coordinates
              </button>
              <button
                onClick={() => setTab('image')}
                className={`rounded-xl px-4 py-1.5 text-sm font-medium ${
                  tab === 'image'
                    ? 'bg-gradient-to-r from-blue-600 to-emerald-500 text-white'
                    : 'bg-white text-slate-700 border border-slate-300'
                }`}
              >
                By Image
              </button>
            </div>

            {tab === 'coords' ? (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-slate-600">Min Lon</label>
                  <input
                    type="number"
                    value={minLon}
                    onChange={e => setMinLon(e.target.value)}
                    className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-600">Min Lat</label>
                  <input
                    type="number"
                    value={minLat}
                    onChange={e => setMinLat(e.target.value)}
                    className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-600">Max Lon</label>
                  <input
                    type="number"
                    value={maxLon}
                    onChange={e => setMaxLon(e.target.value)}
                    className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-600">Max Lat</label>
                  <input
                    type="number"
                    value={maxLat}
                    onChange={e => setMaxLat(e.target.value)}
                    className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
                  />
                </div>
              </div>
            ) : (
              <div
                className="mt-1 w-full rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 hover:bg-slate-100 flex flex-col gap-3 items-center justify-start p-4 cursor-pointer transition"
                onClick={() => document.getElementById('fileInput').click()}
                onDragOver={e => {
                  e.preventDefault();
                  e.stopPropagation();
                  e.currentTarget.classList.add('border-sky-400', 'bg-sky-50');
                }}
                onDragLeave={e => {
                  e.preventDefault();
                  e.stopPropagation();
                  e.currentTarget.classList.remove(
                    'border-sky-400',
                    'bg-sky-50'
                  );
                }}
                onDrop={e => {
                  e.preventDefault();
                  e.stopPropagation();
                  e.currentTarget.classList.remove(
                    'border-sky-400',
                    'bg-sky-50'
                  );
                  const file = e.dataTransfer.files?.[0];
                  if (file) handlePickFile(file);
                }}
              >
                <div
                  className="w-full rounded-lg bg-white border border-slate-200 shadow-sm overflow-auto"
                  style={{ height: uploadPreviewH }}
                >
                  {uploadPreviewUrl ? (
                    <img
                      src={uploadPreviewUrl}
                      alt="selected preview"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="h-full w-full flex flex-col items-center justify-center text-slate-500">
                      <p className="text-sm">
                        Drag &amp; drop image (
                        <span className="font-medium">
                          GeoTIFF / JPEG / PNG
                        </span>
                        ), or
                      </p>
                      <button
                        type="button"
                        className="mt-3 rounded-lg bg-gradient-to-r from-blue-600 to-emerald-500 text-white text-sm px-4 py-2 shadow hover:brightness-110"
                      >
                        Choose File
                      </button>
                    </div>
                  )}
                </div>
                <input
                  id="fileInput"
                  type="file"
                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                  className="hidden"
                  onChange={e => handlePickFile(e.target.files?.[0] || null)}
                />
              </div>
            )}

            <div className="mt-3 grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-600">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={e => setStartDate(e.target.value)}
                  className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
                />
              </div>
              <div>
                <label className="text-xs text-slate-600">End Date</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={e => setEndDate(e.target.value)}
                  className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
                />
              </div>
            </div>

            <button
              onClick={handleSubmit}
              className="mt-6 w-full rounded-2xl bg-gradient-to-r from-blue-600 to-emerald-500 text-white py-2 text-sm font-medium shadow hover:brightness-110"
            >
              Submit
            </button>
            {coordError && (
              <div className="mt-3 text-center text-sm text-red-600">
                {coordError}
              </div>
            )}

            <div className="mt-6">
              <div className="text-sm text-slate-800 mb-2">
                Status:{' '}
                <span
                  className={
                    status === 'done'
                      ? 'text-green-600'
                      : status === 'running'
                      ? 'text-sky-600'
                      : 'text-slate-500'
                  }
                >
                  {status === 'idle'
                    ? 'None'
                    : status === 'running'
                    ? 'Running...'
                    : 'Done'}
                </span>
              </div>
              <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 via-cyan-400 to-emerald-400 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          </div>

          {/* 右卡片 */}
          <div
            className="w-full rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-md flex flex-col"
            style={{ minHeight: rightCardMinH }}
          >
            <div
              className="flex-1 rounded-xl border border-slate-200 bg-slate-100 overflow-auto"
              style={{ height: resultPreviewH }}
            >
              {resultUrl ? (
                <img
                  src={resultUrl}
                  alt="Result"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="h-full w-full flex items-center justify-center text-slate-500 text-sm">
                  Preview area — results will appear here.
                </div>
              )}
            </div>
            {status === 'done' && (
              <div className="mt-4 border-t border-slate-200 pt-4">
                <h3 className="text-sm font-medium text-slate-800 mb-2">
                  Field Statistics
                </h3>
                <div className="text-sm text-slate-600 space-y-1">
                  <p>
                    Number of Fields:{' '}
                    <span className="font-semibold">{fieldCount ?? '-'}</span>
                  </p>
                  <p>
                    Total Area:{' '}
                    <span className="font-semibold">{fieldArea ?? '-'} ha</span>
                  </p>
                </div>
              </div>
            )}

            <div className="flex justify-center mt-4">
              <button
                type="button"
                onClick={handleDownload}
                disabled={!canDownload}
                title={
                  canDownload ? gpkgName || 'result.gpkg' : 'Result not ready'
                }
                className="px-5 py-2 rounded-xl text-sm font-medium bg-gradient-to-r from-blue-600 to-emerald-500 text-white shadow hover:brightness-110"
              >
                Download
              </button>
            </div>
          </div>
        </div>
      </main>

      <footer className="w-full px-[clamp(12px,2vw,24px)] py-6 text-xs text-slate-500 border-t border-slate-200">
        © 2025 Farmland Boundary Detection System
      </footer>
    </div>
  );
}

/* ============================
   简易哈希路由
   ============================ */
function useHashRoute() {
  const [route, setRoute] = useState(window.location.hash || '#/');
  window.onhashchange = () => setRoute(window.location.hash || '#/');
  return route;
}

/* ============================
   应用主入口
   ============================ */
export default function App() {
  const route = useHashRoute();
  return (
    <>
      <GlobalWidthFix />
      {route === '#/use' ? <UsePage /> : <HomePage />}
    </>
  );
}
