# Arquitetura — FIAP Hackathon Threat Modeling

## Diagrama de Arquitetura (Pipeline)

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│   Upload    │────>│   YOLO v8n   │────>│   STRIDE Engine   │────>│   PDF Report │
│  (Imagem)   │     │  (Detector)  │     │   (LLM + RAG)     │     │  (ReportLab) │
└─────────────┘     └──────────────┘     └───────────────────┘     └──────────────┘
      │                    │                       │                       │
      │              DiagramDetections       StrideResult          ThreatModelingReport
      │              (JSON schema)          (JSON schema)            (PDF + JSON)
      │                    │                       │                       │
      ▼                    ▼                       ▼                       ▼
  FastAPI            Ultralytics             OpenAI API              ReportLab
  (main.py)         (detector.py)        (stride_engine.py)    (report_generator.py)
                                                │
                                          ┌─────┴─────┐
                                          │   Local KB │
                                          │  (kb.json) │
                                          └────────────┘
```

---

## Componentes do Sistema

### 1. FastAPI Backend (`backend/main.py`)

**Responsabilidades:**
- Receber upload de imagem via multipart
- Orquestrar pipeline: YOLO → STRIDE → PDF
- Servir PDF gerado para download
- Interface HTML mínima para upload

**Endpoints:**

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/` | Formulário de upload |
| POST | `/analyze` | Pipeline completo (upload → PDF) |
| GET | `/download/{filename}` | Download de PDF gerado |

**Lifecycle:**
- Componentes pesados (YOLO, StrideEngine, ReportGenerator) são inicializados no startup e reutilizados como singletons.

---

### 2. Detector YOLO (`backend/detector.py`)

**Responsabilidades:**
- Carregar modelo YOLO treinado
- Processar imagem (bytes ou path)
- Converter resultados Ultralytics → `DiagramDetections` (schema Pydantic)
- Validar bounding boxes e class labels

**Configuração:**

```python
DetectorConfig(
    weights_path="models/best.pt",
    conf=0.35,        # confidence threshold
    iou=0.45,         # NMS IoU threshold
    imgsz=640,        # inference size
    max_det=300,      # max detections per image
)
```

**Output:** `DiagramDetections` contendo `ImageMeta` + lista de `Detection` (cada uma com id, label, confidence, bbox).

---

### 3. STRIDE Engine (`backend/stride_engine.py`)

**Responsabilidades:**
- Receber `DiagramDetections` do detector
- Construir prompt estruturado com componentes detectados
- Buscar contexto via RAG (KB local + embeddings)
- Chamar LLM (GPT-4.1-mini) para gerar ameaças STRIDE
- Validar output LLM contra schema Pydantic
- Retornar `StrideResult` tipado

**Fluxo interno:**

```
DiagramDetections
    │
    ├─> Mapeamento STRIDE estático (_COMPONENT_TO_STRIDE)
    │
    ├─> RAG: embed query → cosine similarity → top-k KB hits
    │
    ├─> Build prompt (componentes + RAG context + schema example)
    │
    ├─> LLM call (chat completions, temperature=0)
    │
    ├─> Parse JSON → validate via Pydantic → List[Threat]
    │
    └─> StrideResult(threats=[...])
```

**Mapeamento STRIDE por componente:**

Cada classe tem categorias STRIDE pré-definidas (conservadoras). Por exemplo:
- `API Gateway` → todas as 6 categorias
- `Cache` → Tampering, Info Disclosure, DoS
- `User` → Spoofing, Repudiation, Info Disclosure

O LLM gera exatamente N ameaças por componente (padrão: 3).

---

### 4. Base de Conhecimento (`backend/kb.py` + `data/kb.json`)

**Responsabilidades:**
- Carregar e normalizar KB JSON
- Gerenciar embeddings (pre-computados ou lazy via OpenAI)
- Buscar top-k por cosine similarity

**Formato da KB:**

```json
{
  "items": [
    {
      "id": "stride_database_tampering",
      "component": "Database",
      "category": "Tampering",
      "text": "Database tampering: SQL injection, NoSQL injection..."
    }
  ]
}
```

A KB atual contém **60 entradas** cobrindo todas as 14 classes × categorias STRIDE relevantes.

---

### 5. Gerador de PDF (`backend/report_generator.py`)

**Responsabilidades:**
- Receber `ThreatModelingReport` (Pydantic)
- Gerar PDF formatado com ReportLab
- Seções: Capa, Detecções, Tabela STRIDE, Detalhes, Enrichment

---

### 6. Schemas (`backend/schemas.py`)

Todas as estruturas de dados são Pydantic v2 com validação rigorosa:

```
ComponentClass (Enum, 14 classes)
StrideCategory (Enum, 6 categorias)
Severity (Enum: Low/Medium/High/Critical)

Detection → DiagramDetections
Threat → StrideResult
KBHit → EnrichmentResult → EnrichmentBundle
ReportMeta + DiagramDetections + StrideResult → ThreatModelingReport
```

Cross-validation:
- IDs únicos em detecções e ameaças
- Referências cruzadas validadas (threat → detection, enrichment → threat)

---

## Pipeline de ML

### Dataset Sintético

```
Ícones (data/raw_icons/)
    │
    ├─> Randomização (escala, rotação, brilho)
    │
    ├─> Composição em canvas 1280x720
    │   (3-10 componentes, IoU < 0.2)
    │
    ├─> Backgrounds variados (sólido, gradiente, ruído)
    │
    └─> Output: images/*.jpg + labels/*.txt (YOLO format)
```

### Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Modelo base | YOLOv8n (nano) |
| Épocas | 40 (com early stopping) |
| Patience | 10 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | Auto (SGD/Adam) |
| Augmentation | mosaic, flip, HSV, escala |

### Métricas

| Métrica | Meta | Descrição |
|---------|------|-----------|
| **mAP@0.5** | ≥ 0.70 | Métrica principal |
| mAP@0.5:0.95 | ≥ 0.50 | Métrica COCO |
| Precision | ≥ 0.75 | Taxa de verdadeiros positivos |
| Recall | ≥ 0.70 | Cobertura de detecção |

---

## Decisões Arquiteturais

| Decisão | Justificativa |
|---------|---------------|
| YOLOv8n (nano) | Menor complexidade, treino rápido, suficiente para MVP |
| Dataset sintético | Controle total sobre classes e distribuição |
| LLM obrigatório | Qualidade e contextualização das ameaças |
| RAG com KB local | Grounding sem dependência de serviço externo |
| Pydantic v2 | Validação rigorosa entre todas as camadas |
| FastAPI singleton | Evita recarregar modelo YOLO a cada request |
| PDF via ReportLab | Geração confiável sem dependências externas |

---

## Fluxo de Dados

```
                    ┌──────────────────────────────────────────┐
                    │           ThreatModelingReport           │
                    ├──────────────────────────────────────────┤
                    │  meta: ReportMeta                        │
                    │    ├── title                             │
                    │    ├── system_name                       │
                    │    └── generated_at_iso                  │
                    │                                          │
                    │  diagram: DiagramDetections               │
                    │    ├── image: ImageMeta                   │
                    │    └── detections: [Detection]            │
                    │         ├── id, label, confidence         │
                    │         └── bbox: BBox (x1,y1,x2,y2)     │
                    │                                          │
                    │  stride: StrideResult                     │
                    │    └── threats: [Threat]                  │
                    │         ├── threat_id, component_id       │
                    │         ├── category, severity            │
                    │         ├── title, description, impact    │
                    │         └── mitigations: [str]            │
                    │                                          │
                    │  enrichment: EnrichmentBundle (optional)  │
                    │    └── enrichments: [EnrichmentResult]    │
                    │         ├── threat_id                     │
                    │         ├── kb_hits: [KBHit]              │
                    │         └── notes                         │
                    └──────────────────────────────────────────┘
```
