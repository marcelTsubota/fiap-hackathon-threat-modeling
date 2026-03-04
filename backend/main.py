from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import cast

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from backend.detector import DetectorConfig, YoloDetector
from backend.report_generator import ReportGenerator
from backend.schemas import ReportMeta, ThreatModelingReport
from backend.settings import settings
from backend.stride_engine import LLMRequiredError, StrideEngine

app = FastAPI(title="FIAP Threat Modeling MVP")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def _startup() -> None:
    """
    Carrega recursos pesados apenas uma vez.
    - YOLO em CPU é caro para carregar; reutilizamos entre requisições.
    - StrideEngine e ReportGenerator também ficam prontos para uso.
    """
    app.state.detector = YoloDetector(
        DetectorConfig(
            weights_path=settings.MODEL_PATH,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_DUPLICATE_THRESHOLD,
        )
    )
    app.state.engine = StrideEngine()
    app.state.generator = ReportGenerator()


def _get_detector() -> YoloDetector:
    detector = getattr(app.state, "detector", None)
    if detector is None:
        raise RuntimeError("Detector not initialized. Startup event did not run.")
    return cast(YoloDetector, detector)


def _get_engine() -> StrideEngine:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise RuntimeError("StrideEngine not initialized. Startup event did not run.")
    return cast(StrideEngine, engine)


def _get_generator() -> ReportGenerator:
    generator = getattr(app.state, "generator", None)
    if generator is None:
        raise RuntimeError("ReportGenerator not initialized. Startup event did not run.")
    return cast(ReportGenerator, generator)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
    <html><head><title>Threat Modeling</title></head>
    <body>
      <h1>Upload Architecture Diagram</h1>
      <form action="/analyze" enctype="multipart/form-data" method="post">
        <input name="image" type="file" accept="image/*" required>
        <input type="submit" value="Analyze">
      </form>
    </body></html>
    """


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(image: UploadFile = File(...)) -> HTMLResponse:
    # Basic content-type check (content_type can be None)
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    data = await image.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.MAX_IMAGE_MB:
        raise HTTPException(status_code=400, detail="Image too large.")

    try:
        detector = _get_detector()
        diagram = detector.detect_from_bytes(
            data,
            filename=image.filename,
            content_type=image.content_type,
        )

        engine = _get_engine()
        stride_result = engine.analyze(diagram)

    except LLMRequiredError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    meta = ReportMeta(
        title="Threat Modeling Report (STRIDE)",
        system_name=None,
        generated_at_iso=datetime.utcnow().isoformat(),
        version="0.1.0",
    )
    report = ThreatModelingReport(meta=meta, diagram=diagram, stride=stride_result)

    filename = f"report_{uuid.uuid4().hex}.pdf"
    filepath = OUTPUT_DIR / filename

    generator = _get_generator()
    generator.generate(report, str(filepath))

    detections_count = len(diagram.detections)
    threats_count = len(stride_result.threats)

    # Pydantic v2 serialization
    report_json = report.model_dump_json()[:1000]

    html = f"""
    <html><head><title>Analysis Result</title></head><body>
      <h1>Analysis Complete</h1>
      <p>Detections: {detections_count}</p>
      <p>Threats: {threats_count}</p>
      <p><a href="/download/{filename}">Download PDF</a></p>
      <h2>Report (partial JSON)</h2>
      <pre>{report_json}...</pre>
    </body></html>
    """
    return HTMLResponse(content=html)


@app.get("/download/{filename}")
async def download(filename: str) -> FileResponse:
    # Prevent path traversal: keep only basename
    safe = Path(filename).name
    path = OUTPUT_DIR / safe
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)