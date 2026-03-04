# FIAP Hackathon — Modelagem de Ameaças com IA

## Visão Geral

MVP que utiliza IA para interpretar automaticamente diagramas de arquitetura de software, detectar componentes e gerar relatórios de modelagem de ameaças baseados na metodologia **STRIDE**.

### Pipeline

```
Imagem (diagrama) → YOLO (detecção) → JSON → STRIDE Engine (LLM + RAG) → PDF
```

---

## Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Detecção de objetos | YOLOv8n (Ultralytics) |
| Backend API | FastAPI |
| LLM | OpenAI GPT-4.1-mini |
| RAG / Embeddings | text-embedding-3-small + KB local |
| Geração de PDF | ReportLab |
| Validação de dados | Pydantic v2 |
| Processamento de imagem | Pillow, OpenCV |

---

## Estrutura do Projeto

```
├── backend/
│   ├── main.py              # FastAPI (endpoints /analyze, /download)
│   ├── detector.py           # Wrapper YOLO para inferência
│   ├── stride_engine.py      # Engine STRIDE com LLM + RAG
│   ├── kb.py                 # Loader de base de conhecimento local
│   ├── report_generator.py   # Gerador de PDF com ReportLab
│   ├── schemas.py            # Modelos Pydantic (14 classes, STRIDE, Report)
│   └── settings.py           # Configuração via .env
├── scripts/
│   ├── generate_dataset.py   # Gerador de dataset sintético
│   ├── split_dataset.py      # Split treino/validação (80/20)
│   ├── train_yolo.py         # Script de treinamento YOLO
│   └── evaluate.py           # Avaliação e métricas do modelo
├── data/
│   ├── data.yaml             # Configuração YOLO (14 classes)
│   ├── kb.json               # Base de conhecimento STRIDE (60 entradas)
│   ├── raw_icons/            # Ícones para dataset sintético
│   ├── images/               # Dataset gerado (train/val)
│   └── labels/               # Labels YOLO (train/val)
├── models/
│   └── best.pt               # Modelo treinado
├── outputs/                  # PDFs gerados
├── templates/                # Templates HTML (Jinja2)
├── static/                   # CSS/JS estáticos
├── requirements.txt
├── .env.example
└── docs/
    ├── README.md             # Este arquivo
    └── architecture.md       # Documentação de arquitetura
```

---

## Instalação

```bash
# 1. Clonar repositório
git clone https://github.com/marcelTsubota/fiap-hackathon-threat-modeling.git
cd fiap-hackathon-threat-modeling

# 2. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Editar .env e adicionar OPENAI_API_KEY
```

---

## Pipeline Completo (Passo a Passo)

### Fase 1 — Gerar Dataset Sintético

```bash
# Gerar 500 imagens sintéticas (com shapes geométricos)
python scripts/generate_dataset.py --num-images 500 --seed 42

# (Opcional) Para usar ícones reais:
# Colocar PNGs em data/raw_icons/<ClassName>/
# Ex: data/raw_icons/Database/db_01.png
```

### Fase 2 — Split Treino/Validação

```bash
python scripts/split_dataset.py --ratio 0.8 --seed 42
```

### Fase 3 — Treinar YOLO

```bash
# Treino padrão (40 epochs, batch 16, imgsz 640)
python scripts/train_yolo.py

# Com GPU específica
python scripts/train_yolo.py --device 0 --batch 32

# Apenas CPU
python scripts/train_yolo.py --device cpu --batch 8 --epochs 30
```

O melhor modelo é salvo automaticamente em `models/best.pt`.

### Fase 4 — Avaliar Modelo

```bash
python scripts/evaluate.py
```

Métricas geradas:
- **mAP@0.5** (métrica principal)
- mAP@0.5:0.95
- Precision / Recall
- AP por classe
- Confusion matrix (imagem)

### Fase 5 — Executar API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Acessar: http://localhost:8000

Upload de imagem → Detecção → STRIDE → PDF

---

## Classes Detectáveis (14)

| Idx | Classe |
|-----|--------|
| 0 | User |
| 1 | External System |
| 2 | API Gateway |
| 3 | Load Balancer |
| 4 | Web Application |
| 5 | Mobile Application |
| 6 | Application Server |
| 7 | Serverless Function |
| 8 | Database |
| 9 | Cache |
| 10 | Message Queue |
| 11 | File Storage |
| 12 | Authentication Service |
| 13 | Security Service (WAF/Firewall/Shield) |

---

## Metodologia STRIDE

Cada componente detectado é analisado nas 6 categorias:

| Categoria | Descrição |
|---|---|
| **S**poofing | Falsificação de identidade |
| **T**ampering | Alteração não autorizada de dados |
| **R**epudiation | Negação de ações realizadas |
| **I**nformation Disclosure | Vazamento de informações |
| **D**enial of Service | Indisponibilidade do serviço |
| **E**levation of Privilege | Escalonamento de privilégios |

---

## Relatório PDF

O relatório gerado contém:

1. **Capa** com metadados (título, sistema, data)
2. **Lista de detecções** (componente, confiança, bounding box)
3. **Tabela STRIDE** (componente × categoria × severidade)
4. **Detalhes por ameaça** (descrição, impacto, mitigações)
5. **Enriquecimento RAG** (snippets da base de conhecimento)

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `OPENAI_API_KEY` | — | API key do OpenAI (obrigatório) |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Modelo LLM |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings |
| `MODEL_PATH` | `./models/best.pt` | Caminho do modelo YOLO |
| `KB_PATH` | `data/kb.json` | Base de conhecimento |
| `CONFIDENCE_THRESHOLD` | `0.35` | Threshold de confiança |
| `RAG_ENABLED` | `true` | Habilitar RAG |
| `RAG_TOP_K` | `6` | Resultados RAG por query |

---

## Métricas Alvo

| Métrica | Meta |
|---|---|
| mAP@0.5 | ≥ 0.70 |
| Precision | ≥ 0.75 |
| Recall | ≥ 0.70 |
| Latência pipeline | < 30s por diagrama |

---

## Licença

Projeto acadêmico — FIAP Software Security Hackathon.
