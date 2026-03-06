from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.detector import DetectorConfig, YoloDetector
from backend.report_generator import ReportGenerator
from backend.schemas import ReportMeta, StrideResult, ThreatModelingReport
from backend.settings import settings
from backend.stride_engine import LLMRequiredError, StrideEngine

app = FastAPI(title="FIAP Threat Modeling MVP")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(request: Request, image: UploadFile = File(...)):
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

    # Deduplicate threats: group by (title, category, severity) keeping the first occurrence
    seen_keys: set[tuple[str, str, str]] = set()
    unique_threats = []
    for t in stride_result.threats:
        key = (t.title.strip().lower(), t.category.value, t.severity.value)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_threats.append(t)

    # Sort by severity descending (Critical > High > Medium > Low)
    _SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    unique_threats.sort(key=lambda t: _SEVERITY_ORDER.get(t.severity.value, 99))

    stride_result = StrideResult(threats=unique_threats)

    report = ThreatModelingReport(meta=meta, diagram=diagram, stride=stride_result)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"relatorio_stride_{timestamp}.pdf"
    filepath = OUTPUT_DIR / filename

    generator = _get_generator()
    generator.generate(report, str(filepath))

    detections_count = len(diagram.detections)
    threats_count = len(stride_result.threats)

    # Prepare template data
    detections_data = [
        {"id": d.id, "label": d.label.value, "confidence": d.confidence}
        for d in diagram.detections
    ]
    threats_data = [
        {
            "component_label": t.component_label.value,
            "category": t.category.value,
            "severity": t.severity.value,
            "title": t.title,
            "description": t.description,
            "impact": t.impact,
            "mitigations": t.mitigations,
        }
        for t in stride_result.threats
    ]

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "detections_count": detections_count,
            "threats_count": threats_count,
            "pdf_filename": filename,
            "detections": detections_data,
            "threats": threats_data,
        },
    )


@app.get("/download/{filename}")
async def download(filename: str) -> FileResponse:
    # Prevent path traversal: keep only basename
    safe = Path(filename).name
    path = OUTPUT_DIR / safe
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)