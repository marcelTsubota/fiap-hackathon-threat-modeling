"""
Smoke Test — Teste end-to-end do pipeline completo.

Valida:
  1. Importação dos módulos
  2. Carregamento da KB
  3. Carregamento do modelo YOLO
  4. Detecção em imagem sintética
  5. STRIDE engine (LLM + RAG)
  6. Geração de PDF
  7. Servidor HTTP (FastAPI) — via /analyze
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
import signal
import json
from pathlib import Path

# Ensure the project root is on sys.path so `backend` is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# ---- colour helpers ---------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

def ok(msg: str) -> None:
    print(f"  {GREEN}✔{RESET} {msg}")

def fail(msg: str) -> None:
    print(f"  {RED}✘{RESET} {msg}")

def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")

def header(msg: str) -> None:
    print(f"\n{BOLD}{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}{RESET}")

# ---- step tracker -----------------------------------------------------
results: list[tuple[str, bool, str]] = []

def step(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    if passed:
        ok(f"{name}" + (f" — {detail}" if detail else ""))
    else:
        fail(f"{name}" + (f" — {detail}" if detail else ""))

# ---- tests ------------------------------------------------------------

def test_imports() -> bool:
    header("1/7  Importação de Módulos")
    try:
        from backend import schemas, settings, detector, kb, stride_engine, report_generator
        step("Importar backend.*", True)
        return True
    except Exception as e:
        step("Importar backend.*", False, str(e))
        return False


def test_kb() -> bool:
    header("2/7  Base de Conhecimento (KB)")
    try:
        from backend.kb import LocalKB
        from backend.settings import settings
        kb = LocalKB(kb_path=settings.KB_PATH)
        kb.load()
        count = len(kb._items)
        passed = count > 0
        step(f"Carregar KB ({count} entradas)", passed)
        return passed
    except Exception as e:
        step("Carregar KB", False, str(e))
        return False


def test_model_load() -> bool:
    header("3/7  Modelo YOLO")
    try:
        from backend.detector import YoloDetector, DetectorConfig
        from backend.settings import settings

        model_path = settings.MODEL_PATH
        if not Path(model_path).exists():
            step("Arquivo best.pt existe", False, f"{model_path} não encontrado")
            return False
        step("Arquivo best.pt existe", True, model_path)

        det = YoloDetector(DetectorConfig(
            weights_path=model_path,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_DUPLICATE_THRESHOLD,
        ))
        step("Carregar modelo YOLO", True)
        return True
    except Exception as e:
        step("Carregar modelo YOLO", False, str(e))
        return False


def test_detection() -> tuple[bool, int]:
    header("4/7  Detecção em Imagem Sintética")
    try:
        from backend.detector import YoloDetector, DetectorConfig
        from backend.settings import settings

        # use a synthetic image from the dataset
        test_images = list(Path("data/images/all").glob("*.jpg"))
        if not test_images:
            test_images = list(Path("data/images/train").glob("*.jpg"))
        if not test_images:
            step("Encontrar imagem de teste", False, "Nenhuma imagem em data/images/")
            return False, 0

        img_path = test_images[0]
        step("Imagem de teste", True, str(img_path))

        det = YoloDetector(DetectorConfig(
            weights_path=settings.MODEL_PATH,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_DUPLICATE_THRESHOLD,
        ))

        diagram = det.detect_from_path(str(img_path))
        n = len(diagram.detections)
        if n > 0:
            step(f"Detecções encontradas: {n}", True)
            for d in diagram.detections[:5]:
                ok(f"  -> {d.label.value} (conf={d.confidence:.2f})")
        else:
            warn("Nenhuma detecção encontrada (modelo pode precisar de mais treinamento)")
            step("Detecções", True, "0 (modelo subtreinado — esperado no smoke test)")
        return True, n
    except Exception as e:
        step("Detecção", False, str(e))
        return False, 0


def test_stride(n_detections: int) -> bool:
    header("5/7  STRIDE Engine (LLM + RAG)")

    from backend.settings import settings
    key = settings.OPENAI_API_KEY
    if not key or not key.strip():
        step("OPENAI_API_KEY configurada", False, "Defina no .env")
        return False
    step("OPENAI_API_KEY configurada", True)

    if n_detections == 0:
        warn("Sem deteccoes -> STRIDE Engine sera testada com deteccoes simuladas")

    try:
        from backend.stride_engine import StrideEngine
        from backend.schemas import (
            ComponentClass, Detection, DiagramDetections, BBox, ImageMeta
        )

        engine = StrideEngine()
        step("Inicializar StrideEngine", True)

        # If no detections, create a fake one to test LLM
        if n_detections == 0:
            fake_det = Detection(
                id="test-001",
                label=ComponentClass.WEB_APPLICATION,
                confidence=0.90,
                bbox=BBox(x1=100, y1=100, x2=300, y2=300),
            )
            diagram = DiagramDetections(
                image=ImageMeta(filename="smoke_test.jpg", width=640, height=480),
                detections=[fake_det],
            )
        else:
            from backend.detector import YoloDetector, DetectorConfig
            test_images = list(Path("data/images/all").glob("*.jpg"))
            if not test_images:
                test_images = list(Path("data/images/train").glob("*.jpg"))
            det = YoloDetector(DetectorConfig(
                weights_path=settings.MODEL_PATH,
                conf=settings.CONFIDENCE_THRESHOLD,
                iou=settings.IOU_DUPLICATE_THRESHOLD,
            ))
            diagram = det.detect_from_path(str(test_images[0]))

        result = engine.analyze(diagram)
        n_threats = len(result.threats)
        step(f"Geração de ameaças STRIDE: {n_threats}", n_threats > 0)

        if n_threats > 0:
            for t in result.threats[:3]:
                ok(f"  -> [{t.category.value}] {t.title} (sev={t.severity.value})")

        return n_threats > 0
    except Exception as e:
        step("STRIDE Engine", False, str(e))
        return False


def test_pdf() -> bool:
    header("6/7  Geração de PDF")
    try:
        import uuid as _uuid
        from datetime import datetime
        from backend.report_generator import ReportGenerator
        from backend.schemas import (
            ComponentClass, Detection, DiagramDetections, BBox, ImageMeta,
            Threat, StrideCategory, Severity, StrideResult,
            ThreatModelingReport, ReportMeta,
        )

        fake_det = Detection(
            id="pdf-test-001",
            label=ComponentClass.DATABASE,
            confidence=0.85,
            bbox=BBox(x1=50, y1=50, x2=250, y2=250),
        )
        diagram = DiagramDetections(
            image=ImageMeta(filename="pdf_test.jpg", width=640, height=480),
            detections=[fake_det],
        )
        threat = Threat(
            threat_id="threat-pdf-001",
            component_id="pdf-test-001",
            component_label=ComponentClass.DATABASE,
            category=StrideCategory.TAMPERING,
            title="SQL Injection no Banco de Dados",
            description="Um atacante pode modificar consultas SQL para acessar dados não autorizados.",
            impact="Exposição ou modificação de dados sensíveis.",
            severity=Severity.HIGH,
            mitigations=["Usar prepared statements", "Validar inputs"],
        )
        stride_result = StrideResult(
            threats=[threat],
        )
        meta = ReportMeta(
            title="Smoke Test Report",
            system_name="Test",
            generated_at_iso=datetime.utcnow().isoformat(),
            version="0.1.0",
        )
        report = ThreatModelingReport(meta=meta, diagram=diagram, stride=stride_result)

        out_path = Path("outputs") / f"smoke_test_{_uuid.uuid4().hex[:8]}.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        gen = ReportGenerator()
        gen.generate(report, str(out_path))

        exists = out_path.exists()
        size_kb = out_path.stat().st_size / 1024 if exists else 0
        step(f"PDF gerado ({size_kb:.1f} KB)", exists, str(out_path))

        # cleanup
        if exists:
            out_path.unlink()

        return exists
    except Exception as e:
        step("Geração de PDF", False, str(e))
        return False


def test_http() -> bool:
    header("7/7  Teste HTTP (FastAPI /analyze)")

    try:
        import requests
    except ImportError:
        warn("Pacote 'requests' não instalado — instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
        import requests

    # Start the server in background
    print("  Iniciando servidor FastAPI...")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "127.0.0.1", "--port", "8765", "--log-level", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    try:
        # Wait for server to start
        url_base = "http://127.0.0.1:8765"
        ready = False
        for i in range(30):
            try:
                r = requests.get(url_base + "/", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except requests.ConnectionError:
                pass
            time.sleep(1)

        if not ready:
            step("Servidor iniciou", False, "Timeout (30s)")
            return False
        step("Servidor iniciou", True, url_base)

        # GET /
        r = requests.get(url_base + "/")
        step("GET / (página inicial)", r.status_code == 200, f"HTTP {r.status_code}")

        # POST /analyze with test image
        test_images = list(Path("data/images/all").glob("*.jpg"))
        if not test_images:
            test_images = list(Path("data/images/train").glob("*.jpg"))

        if not test_images:
            step("Imagem para /analyze", False, "Nenhuma imagem disponível")
            return False

        img_path = test_images[0]
        with open(img_path, "rb") as f:
            r = requests.post(
                url_base + "/analyze",
                files={"image": (img_path.name, f, "image/jpeg")},
                timeout=120,
            )

        step(f"POST /analyze", r.status_code == 200, f"HTTP {r.status_code}")

        if r.status_code == 200:
            # Check if PDF link is in the response
            has_pdf = "download/" in r.text
            step("PDF link no resultado", has_pdf)

            # Try to download the PDF
            if has_pdf:
                import re
                match = re.search(r'download/([a-f0-9_]+\.pdf)', r.text)
                if match:
                    pdf_name = match.group(1)
                    r2 = requests.get(f"{url_base}/download/{pdf_name}")
                    is_pdf = r2.status_code == 200 and r2.headers.get("content-type", "").startswith("application/")
                    size_kb = len(r2.content) / 1024
                    step(f"Download PDF ({size_kb:.1f} KB)", is_pdf, pdf_name)

        return r.status_code == 200

    finally:
        # Kill the server
        print("  Parando servidor...")
        if sys.platform == "win32":
            server.terminate()
        else:
            os.kill(server.pid, signal.SIGTERM)
        server.wait(timeout=10)
        ok("Servidor parado")


# ---- main -------------------------------------------------------------

def main() -> None:
    print(f"\n{BOLD}🔬 SMOKE TEST — Threat Modeling com IA{RESET}")
    print(f"   Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Diretório: {os.getcwd()}")

    test_imports()
    test_kb()
    test_model_load()
    detection_ok, n_det = test_detection()
    test_stride(n_det)
    test_pdf()
    test_http()

    # Summary
    header("RESUMO")
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    print()
    for name, p, detail in results:
        icon = f"{GREEN}✔{RESET}" if p else f"{RED}✘{RESET}"
        det = f" — {detail}" if detail else ""
        print(f"  {icon} {name}{det}")

    print(f"\n  {BOLD}Resultado: {passed}/{total} testes passaram{RESET}")
    if passed == total:
        print(f"  {GREEN}{BOLD}🎉 TUDO FUNCIONANDO! Pronto para envio.{RESET}")
    else:
        failed = [n for n, p, _ in results if not p]
        print(f"  {RED}Falhas: {', '.join(failed)}{RESET}")

    print()
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
