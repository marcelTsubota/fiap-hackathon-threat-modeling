# 📖 Manual de Instruções — Threat Modeling com IA

**FIAP Hackathon — Software Security · Modelagem de Ameaças com Inteligência Artificial**

> Versão 0.1.0 · Atualizado em: Março 2026

---

## Sumário

1. [Visão Geral do Sistema](#1-visão-geral-do-sistema)
2. [Arquitetura e Fluxo de Dados](#2-arquitetura-e-fluxo-de-dados)
3. [Pré-requisitos](#3-pré-requisitos)
4. [Instalação e Configuração](#4-instalação-e-configuração)
5. [Pipeline de Machine Learning](#5-pipeline-de-machine-learning)
   - 5.1 [Geração do Dataset Sintético](#51-geração-do-dataset-sintético)
   - 5.2 [Divisão Train/Val](#52-divisão-trainval)
   - 5.3 [Treinamento do Modelo YOLO](#53-treinamento-do-modelo-yolo)
   - 5.4 [Avaliação do Modelo](#54-avaliação-do-modelo)
6. [Servidor Web (API)](#6-servidor-web-api)
   - 6.1 [Iniciar o Servidor](#61-iniciar-o-servidor)
   - 6.2 [Endpoints da API](#62-endpoints-da-api)
   - 6.3 [Fluxo do Usuário na Interface Web](#63-fluxo-do-usuário-na-interface-web)
7. [Componentes do Backend](#7-componentes-do-backend)
   - 7.1 [Detector YOLO (`detector.py`)](#71-detector-yolo-detectorpy)
   - 7.2 [Motor STRIDE (`stride_engine.py`)](#72-motor-stride-stride_enginepy)
   - 7.3 [Gerador de Relatório PDF (`report_generator.py`)](#73-gerador-de-relatório-pdf-report_generatorpy)
   - 7.4 [Base de Conhecimento (`kb.py` + `kb.json`)](#74-base-de-conhecimento-kbpy--kbjson)
   - 7.5 [Schemas Pydantic (`schemas.py`)](#75-schemas-pydantic-schemaspy)
   - 7.6 [Configurações (`settings.py`)](#76-configurações-settingspy)
8. [As 14 Classes Fixas](#8-as-14-classes-fixas)
9. [Metodologia STRIDE](#9-metodologia-stride)
10. [Base de Conhecimento (RAG)](#10-base-de-conhecimento-rag)
11. [Variáveis de Ambiente](#11-variáveis-de-ambiente)
12. [Estrutura de Diretórios](#12-estrutura-de-diretórios)
13. [Solução de Problemas](#13-solução-de-problemas)
14. [Glossário](#14-glossário)

---

## 1. Visão Geral do Sistema

O sistema é um **MVP (Minimum Viable Product)** que automatiza a **modelagem de ameaças** a partir de diagramas de arquitetura de software. Ele combina duas tecnologias de IA:

| Tecnologia | Função |
|---|---|
| **YOLOv8n** (Ultralytics) | Detecta componentes arquiteturais em imagens de diagramas |
| **GPT-4.1-mini** (OpenAI) | Gera ameaças STRIDE para cada componente detectado |

### Como funciona (resumo):

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Upload de   │ ──→ │  Detecção    │ ──→ │  Análise     │ ──→ │  Relatório   │
│  Diagrama    │     │  YOLO        │     │  STRIDE/LLM  │     │  PDF         │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
   (imagem)          (14 classes)         (ameaças/mitigações)   (download)
```

**Entrada:** Imagem de diagrama de arquitetura (PNG, JPG, etc.)  
**Saída:** Relatório PDF com ameaças identificadas, classificadas por categoria STRIDE, com severidade e mitigações sugeridas.

---

## 2. Arquitetura e Fluxo de Dados

```
                          ┌─────────────────────────┐
                          │    Interface Web (UI)    │
                          │    templates/ + static/  │
                          └───────────┬─────────────┘
                                      │ POST /analyze (multipart)
                                      ▼
                          ┌─────────────────────────┐
                          │    FastAPI (main.py)     │
                          │    Orquestrador          │
                          └──┬──────────┬──────────┬┘
                             │          │          │
                    ┌────────▼──┐  ┌────▼────┐  ┌──▼─────────────┐
                    │ detector  │  │ stride  │  │ report         │
                    │ .py       │  │ _engine │  │ _generator.py  │
                    │           │  │ .py     │  │                │
                    │ YOLOv8n   │  │ LLM+RAG │  │ ReportLab PDF  │
                    └────────┬──┘  └────┬────┘  └──┬─────────────┘
                             │          │          │
                    ┌────────▼──┐  ┌────▼────┐    │
                    │ models/   │  │ OpenAI  │    │
                    │ best.pt   │  │ API     │    │
                    └───────────┘  └────┬────┘    │
                                   ┌────▼────┐    │
                                   │ data/   │    │
                                   │ kb.json │    │
                                   └─────────┘    │
                                             ┌────▼────┐
                                             │outputs/ │
                                             │ *.pdf   │
                                             └─────────┘
```

### Fluxo detalhado:

1. **Upload:** Usuário envia imagem do diagrama via formulário web
2. **Validação:** FastAPI valida tipo MIME e tamanho (máx. 10 MB)
3. **Detecção YOLO:** Modelo identifica componentes arquiteturais com bounding boxes e confiança
4. **Construção RAG:** Motor STRIDE busca contexto relevante na base de conhecimento local (60 entradas, similaridade por cosseno via embeddings)
5. **Análise LLM:** GPT-4.1-mini recebe componentes detectados + contexto RAG e gera ameaças STRIDE em JSON
6. **Validação:** Resposta do LLM é parseada e validada contra schemas Pydantic rigorosos
7. **Geração PDF:** ReportLab gera relatório com capa, tabelas de detecções, tabelas STRIDE, cards detalhados de cada ameaça
8. **Resultado:** Página HTML com resumo visual + link para download do PDF

---

## 3. Pré-requisitos

### Software necessário:

| Software | Versão Mínima | Propósito |
|---|---|---|
| Python | 3.10+ (recomendado 3.11) | Linguagem principal |
| pip | 22+ | Gerenciador de pacotes |
| Git | 2.30+ | Controle de versão |

### Contas e chaves:

| Serviço | Necessário? | Para quê |
|---|---|---|
| OpenAI API Key | **Sim (obrigatório)** | LLM (GPT-4.1-mini) + Embeddings (text-embedding-3-small) |

> ⚠️ **IMPORTANTE:** O sistema **não funciona sem** a chave `OPENAI_API_KEY`. A análise STRIDE depende do LLM para gerar ameaças e do modelo de embeddings para o RAG.

### Hardware recomendado:

- **CPU:** Qualquer processador moderno (o modelo YOLO roda em CPU)
- **RAM:** 4 GB mínimo (8 GB recomendado para treinamento)
- **GPU:** Opcional (acelera treinamento, mas não é necessário para inferência)
- **Disco:** ~2 GB (modelo + dataset + dependências)

---

## 4. Instalação e Configuração

### Passo 1 — Clonar o repositório

```bash
git clone https://github.com/marcelTsubota/fiap-hackathon-threat-modeling.git
cd fiap-hackathon-threat-modeling
```

### Passo 2 — Criar ambiente virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Passo 3 — Instalar dependências

```bash
pip install -r requirements.txt
```

**Bibliotecas instaladas:**

| Pacote | Função |
|---|---|
| `fastapi` | Framework web assíncrono |
| `uvicorn` | Servidor ASGI |
| `ultralytics` | YOLO v8 (detecção de objetos) |
| `openai` | SDK da API OpenAI |
| `reportlab` | Geração de PDF |
| `pillow` | Processamento de imagens |
| `opencv-python` | Operações de visão computacional |
| `numpy` | Computação numérica |
| `pydantic` / `pydantic-settings` | Validação de dados e configurações |
| `jinja2` | Templates HTML |
| `python-dotenv` | Variáveis de ambiente |
| `python-multipart` | Upload de arquivos |
| `httpx` | Cliente HTTP assíncrono |
| `rich` | Formatação de terminal |

### Passo 4 — Configurar variáveis de ambiente

```bash
# Copiar template
cp .env.example .env

# Editar e inserir sua chave OpenAI
# No arquivo .env, alterar:
OPENAI_API_KEY=sk-sua-chave-aqui
```

### Passo 5 — Verificar instalação

```bash
python -c "from backend.schemas import ComponentClass; print('OK:', len(list(ComponentClass)), 'classes')"
# Saída esperada: OK: 14 classes
```

---

## 5. Pipeline de Machine Learning

O pipeline ML transforma imagens de diagramas em detecções de componentes. Ele deve ser executado na seguinte ordem:

```
generate_dataset.py → split_dataset.py → train_yolo.py → evaluate.py
```

### 5.1 Geração do Dataset Sintético

O script gera imagens sintéticas de diagramas de arquitetura com anotações YOLO automáticas.

```bash
python scripts/generate_dataset.py --num-images 500 --seed 42
```

**O que acontece:**
- Cria 500 imagens de 1280×720 pixels em `data/images/all/`
- Gera labels YOLO correspondentes em `data/labels/all/`
- Cada imagem contém 3–10 componentes arquiteturais aleatórios
- Aplica augmentations: escala (0.6x–1.4x), rotação (±15°), brilho (0.8–1.2x)
- Controla sobreposição entre componentes (IoU < 0.2)
- Fundos variados: sólido, gradiente, com ruído

**Parâmetros customizáveis:**

| Parâmetro | Default | Descrição |
|---|---|---|
| `--num-images` | 500 | Quantidade de imagens |
| `--output-dir` | data | Diretório de saída |
| `--icons` | data/raw_icons | Pasta com ícones reais (opcional) |
| `--width` | 1280 | Largura do canvas |
| `--height` | 720 | Altura do canvas |
| `--min-components` | 3 | Mínimo de componentes por imagem |
| `--max-components` | 10 | Máximo de componentes por imagem |
| `--seed` | 42 | Semente aleatória para reprodutibilidade |

**Sobre ícones:**
- **Com ícones reais:** Coloque PNGs em `data/raw_icons/<NomeDaClasse>/`. Ex: `data/raw_icons/Database/db_aws.png`
- **Sem ícones:** O sistema usa formas geométricas coloridas com rótulos (círculos, retângulos, diamantes, hexágonos) — funcional para MVP

**Formato do label YOLO** (por linha):
```
<class_idx> <x_center> <y_center> <width> <height>
```
Valores normalizados entre 0 e 1. Exemplo: `8 0.553 0.412 0.078 0.139` = Database no centro-direito da imagem.

### 5.2 Divisão Train/Val

Divide as imagens geradas em conjuntos de treino (80%) e validação (20%).

```bash
python scripts/split_dataset.py --ratio 0.8 --seed 42
```

**O que acontece:**
- Lê imagens de `data/images/all/` e labels de `data/labels/all/`
- Embaralha deterministicamente (com seed)
- Copia 80% para `data/images/train/` + `data/labels/train/`
- Copia 20% para `data/images/val/` + `data/labels/val/`

**Parâmetros:**

| Parâmetro | Default | Descrição |
|---|---|---|
| `--data-dir` | data | Diretório raiz do dataset |
| `--ratio` | 0.8 | Proporção para treino (0.0–1.0) |
| `--seed` | 42 | Semente aleatória |

### 5.3 Treinamento do Modelo YOLO

Treina o modelo YOLOv8n para detectar as 14 classes de componentes.

```bash
python scripts/train_yolo.py --epochs 40 --batch 16
```

**O que acontece:**
- Carrega o modelo base `yolov8n.pt` (pré-treinado no COCO)
- Realiza transfer learning para as 14 classes customizadas
- Treina por 40 épocas com early stopping (patience=10)
- Salva os melhores pesos automaticamente em `models/best.pt`
- Logs de treinamento em `runs/detect/`

**Hiperparâmetros configurados:**

| Parâmetro | Valor | Descrição |
|---|---|---|
| Épocas | 40 | Iterações completas sobre o dataset |
| Batch size | 16 | Imagens por lote |
| Image size | 640px | Dimensão de entrada |
| Patience | 10 | Épocas sem melhora antes de parar |
| Optimizer | auto (AdamW) | Otimizador automático |
| Learning rate | 0.01 | Taxa de aprendizado inicial |
| Mosaic | 1.0 | Augmentation de mosaico |
| Flip LR | 0.5 | Espelhamento horizontal |
| Rotação | ±15° | Variação de ângulo |

**Parâmetros do CLI:**

| Parâmetro | Default | Descrição |
|---|---|---|
| `--model` | yolov8n.pt | Modelo base |
| `--data` | data/data.yaml | Configuração do dataset |
| `--epochs` | 40 | Número de épocas |
| `--batch` | 16 | Tamanho do batch |
| `--imgsz` | 640 | Tamanho da imagem |
| `--patience` | 10 | Early stopping |
| `--device` | auto | `cpu`, `0` (GPU), ou `0,1` (multi-GPU) |
| `--resume` | false | Retomar de checkpoint |

**Saída:**
```
✅ Best weights copied to: models/best.pt
```

### 5.4 Avaliação do Modelo

Avalia o modelo treinado e gera métricas detalhadas.

```bash
python scripts/evaluate.py --model models/best.pt
```

**O que acontece:**
- Executa validação completa no conjunto `val`
- Calcula métricas: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- Gera AP@0.5 por classe (com barra visual)
- Salva métricas em JSON (`runs/evaluate/val/metrics.json`)
- Roda inferência em amostras e salva imagens anotadas

**Critérios de qualidade:**

| mAP@0.5 | Veredicto |
|---|---|
| ≥ 0.70 | ✅ Pronto para produção |
| ≥ 0.50 | ⚠️ Aceitável (pode ser melhorado) |
| < 0.50 | ❌ Precisa de melhoria |

**Meta do projeto:** mAP@0.5 ≥ 0.70

---

## 6. Servidor Web (API)

### 6.1 Iniciar o Servidor

```bash
# Desenvolvimento (com auto-reload)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Produção
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Acesse: **http://localhost:8000**

### 6.2 Endpoints da API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Página inicial com formulário de upload |
| `POST` | `/analyze` | Envia imagem e retorna resultado da análise |
| `GET` | `/download/{filename}` | Download do relatório PDF gerado |
| `GET` | `/docs` | Documentação Swagger automática (FastAPI) |
| `GET` | `/redoc` | Documentação ReDoc alternativa |

#### Detalhes do `POST /analyze`:

**Request:**
- Content-Type: `multipart/form-data`
- Campo: `image` (arquivo de imagem)
- Tamanho máximo: 10 MB
- Tipos aceitos: `image/png`, `image/jpeg`, `image/webp`, etc.

**Response (sucesso):**
- Página HTML com resultado renderizado (tabela de detecções, ameaças STRIDE, link para PDF)

**Response (erro):**
- `400` — Arquivo não é imagem ou excede tamanho máximo
- `500` — Erro interno (modelo YOLO, LLM, etc.)

#### Exemplo via cURL:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "image=@meu_diagrama.png" \
  -o resultado.html
```

### 6.3 Fluxo do Usuário na Interface Web

1. **Acesso:** Navegar para `http://localhost:8000`
2. **Upload:** Arrastar e soltar imagem do diagrama (ou clicar para selecionar)
3. **Pré-visualização:** Imagem aparece no formulário antes do envio
4. **Analisar:** Clicar no botão "🔍 Analisar Diagrama"
5. **Aguardar:** Barra de carregamento enquanto o pipeline processa
6. **Resultado:** Página com:
   - Cards de resumo (total de componentes, total de ameaças)
   - Tabela de componentes detectados com barras de confiança
   - Tabela de ameaças STRIDE com badges coloridos por categoria e severidade
   - Cards detalhados de cada ameaça com mitigações
   - Botão de download do PDF
7. **Download:** Clicar em "📄 Download PDF" para baixar o relatório

---

## 7. Componentes do Backend

### 7.1 Detector YOLO (`detector.py`)

**Responsabilidade:** Executar inferência YOLO em imagens e converter resultados para o schema interno.

**Classes principais:**

| Classe/Função | Descrição |
|---|---|
| `YoloDetector` | Wrapper principal — carrega modelo e executa inferência |
| `DetectorConfig` | Configuração: path do modelo, conf threshold, IoU NMS, imgsz |
| `detect_from_bytes()` | Inferência a partir de bytes (upload web) |
| `detect_from_path()` | Inferência a partir de arquivo local |

**Fluxo interno:**
1. Recebe imagem (bytes ou path)
2. Converte para RGB via Pillow
3. Converte para numpy array (H, W, 3)
4. Executa `model.predict()` do Ultralytics
5. Converte bounding boxes para schema `Detection`
6. Valida labels contra `ComponentClass` (14 classes fixas)
7. Retorna `DiagramDetections` (metadados da imagem + lista de detecções)

**Parâmetros de inferência:**

| Parâmetro | Default | Descrição |
|---|---|---|
| `conf` | 0.35 | Confiança mínima para manter detecção |
| `iou` | 0.50 | Threshold IoU para NMS (supressão de não-máximos) |
| `imgsz` | 640 | Tamanho de entrada do modelo |
| `max_det` | 300 | Máximo de detecções por imagem |

### 7.2 Motor STRIDE (`stride_engine.py`)

**Responsabilidade:** Gerar ameaças STRIDE usando LLM (GPT-4.1-mini) com contexto RAG da base de conhecimento local.

**Classes principais:**

| Classe/Função | Descrição |
|---|---|
| `StrideEngine` | Classe principal — orquestra prompt, RAG, chamada LLM, parsing |
| `StrideEngineConfig` | Configuração: modelo LLM, modelo embedding, path KB, top_k |
| `_LocalKB` | Loader interno da base de conhecimento |

**Fluxo interno:**

1. **Construção RAG:** Para cada componente detectado, busca entradas relevantes na KB via similaridade de cosseno (embeddings `text-embedding-3-small`)
2. **Construção do Prompt:** Monta prompt estruturado com:
   - Instruções de segurança (role: senior security engineer)
   - Regras estritas de output JSON
   - Lista de componentes detectados com STRIDE categories permitidas
   - Contexto RAG (snippets da KB)
3. **Chamada LLM:** `chat.completions.create()` com `temperature=0`
4. **Parsing JSON:** Parse estrito + fallback para extração de JSON
5. **Validação Pydantic:** Cada ameaça é validada contra o schema `Threat`
6. **Verificação de integridade:** Valida que `component_id` referencia detecções reais

**Mapeamento Componente → STRIDE:**

O sistema usa um mapeamento pré-definido (`_COMPONENT_TO_STRIDE`) para informar ao LLM quais categorias STRIDE são relevantes para cada tipo de componente. O LLM deve gerar exatamente **3 ameaças por componente**.

### 7.3 Gerador de Relatório PDF (`report_generator.py`)

**Responsabilidade:** Gerar PDF profissional a partir do `ThreatModelingReport` usando ReportLab.

**Seções do PDF:**

1. **Capa:** Título, nome do sistema (opcional), timestamp
2. **Tabela de Detecções:** ID, Label, Confiança
3. **Tabela STRIDE:** Componente, Categoria, Severidade, Título, Confiança
4. **Cards de Ameaças:** Para cada ameaça:
   - Título
   - Componente, Categoria, Severidade
   - Descrição detalhada
   - Impacto
   - Lista de mitigações (bullets)
5. **Enrichment (RAG):** Se disponível, mostra hits da KB por ameaça

### 7.4 Base de Conhecimento (`kb.py` + `kb.json`)

**Responsabilidade:** Armazenar e recuperar conhecimento sobre ameaças STRIDE para grounding do LLM.

**Formato do `kb.json`:**
```json
{
  "items": [
    {
      "id": "stride_user_spoofing",
      "component": "User",
      "category": "Spoofing",
      "text": "Descrição detalhada da ameaça, vetores de ataque e mitigações..."
    }
  ]
}
```

**Características:**
- 60 entradas cobrindo todas as 14 classes × categorias STRIDE relevantes
- Cada entrada contém ameaças específicas, técnicas de ataque e mitigações
- Embeddings são calculados sob demanda (lazy) usando `text-embedding-3-small`
- Recuperação por top-k similaridade de cosseno (k=6, min_score=0.25)
- Cache de embeddings em memória para evitar chamadas repetidas

### 7.5 Schemas Pydantic (`schemas.py`)

**Responsabilidade:** Definir todos os modelos de dados com validação estrita (Pydantic v2).

**Modelos principais:**

| Schema | Descrição |
|---|---|
| `ComponentClass` | Enum com as 14 classes fixas |
| `StrideCategory` | Enum STRIDE: Spoofing, Tampering, Repudiation, Information Disclosure, DoS, EoP |
| `Severity` | Enum: Low, Medium, High, Critical |
| `BBox` | Bounding box (x1, y1, x2, y2) com validação geométrica |
| `Detection` | Detecção individual: id, label, confiança, bbox |
| `DiagramDetections` | Imagem + lista de detecções (IDs únicos) |
| `Threat` | Ameaça STRIDE: componente, categoria, severidade, título, descrição, impacto, mitigações |
| `StrideResult` | Lista de ameaças (IDs únicos) |
| `KBHit` | Resultado de busca na KB (source, key, score, snippet) |
| `LLMUsage` | Telemetria de uso do LLM (tokens) |
| `ThreatModelingReport` | Modelo consolidado com validação cruzada entre detecções, ameaças e enrichments |

**Validações cruzadas:**
- Todas as ameaças DEVEM referenciar `component_id` válidos
- Todos os enrichments DEVEM referenciar `threat_id` válidos
- IDs de detecção e ameaças devem ser únicos

### 7.6 Configurações (`settings.py`)

**Responsabilidade:** Gerenciar configurações via variáveis de ambiente (`.env`) usando `pydantic-settings`.

Todas as configurações têm valores default e podem ser sobrescritas pelo arquivo `.env`:

```python
MODEL_PATH = "./models/best.pt"     # Path do modelo YOLO
KB_PATH = "data/kb.json"            # Path da base de conhecimento
OPENAI_LLM_MODEL = "gpt-4.1-mini"  # Modelo LLM
CONFIDENCE_THRESHOLD = 0.35         # Threshold de confiança do YOLO
```

---

## 8. As 14 Classes Fixas

O sistema reconhece **exatamente 14 tipos** de componentes arquiteturais. Esta lista é **imutável** — o modelo YOLO, a KB e o motor STRIDE estão todos alinhados.

| Índice | Classe | Forma (sintético) | Cor |
|---|---|---|---|
| 0 | User | Círculo | Azul |
| 1 | External System | Retângulo | Roxo |
| 2 | API Gateway | Retângulo | Laranja |
| 3 | Load Balancer | Retângulo | Verde |
| 4 | Web Application | Retângulo | Azul claro |
| 5 | Mobile Application | Círculo | Roxo |
| 6 | Application Server | Retângulo | Vermelho |
| 7 | Serverless Function | Diamante | Amarelo |
| 8 | Database | Retângulo arredondado | Teal |
| 9 | Cache | Retângulo arredondado | Rosa |
| 10 | Message Queue | Diamante | Laranja |
| 11 | File Storage | Retângulo | Cinza |
| 12 | Authentication Service | Hexágono | Verde claro |
| 13 | Security Service (WAF/Firewall/Shield) | Hexágono | Vermelho |

> ⚠️ **NÃO reordene, renomeie ou adicione classes.** Os labels YOLO são baseados em índice.

---

## 9. Metodologia STRIDE

O sistema utiliza a metodologia **STRIDE** da Microsoft para categorizar ameaças:

| Categoria | Sigla | Descrição | Pergunta-chave |
|---|---|---|---|
| **S**poofing | S | Falsificação de identidade | "Alguém pode se passar por outro?" |
| **T**ampering | T | Adulteração de dados | "Os dados podem ser modificados indevidamente?" |
| **R**epudiation | R | Repúdio (negar ações) | "As ações podem ser rastreadas?" |
| **I**nformation Disclosure | I | Divulgação de informações | "Dados sensíveis podem vazar?" |
| **D**enial of Service | D | Negação de serviço | "O serviço pode ser derrubado?" |
| **E**levation of Privilege | E | Elevação de privilégio | "Um usuário pode obter permissões indevidas?" |

Cada componente é mapeado para as categorias STRIDE mais relevantes. O LLM então gera **3 ameaças por componente** com:
- Título descritivo
- Descrição técnica
- Impacto no negócio
- Severidade (Low/Medium/High/Critical)
- Mitigações específicas e acionáveis

---

## 10. Base de Conhecimento (RAG)

O sistema implementa **Retrieval-Augmented Generation (RAG)** para melhorar a qualidade das ameaças geradas:

### Como funciona:

1. **Indexação:** Ao iniciar, carrega `data/kb.json` (60 entradas)
2. **Embedding:** Para cada entrada, gera embedding via `text-embedding-3-small` (lazy, sob demanda)
3. **Busca:** Para cada componente detectado, busca as entradas mais relevantes por similaridade de cosseno
4. **Contexto:** Snippets recuperados são injetados no prompt do LLM como contexto de grounding
5. **Geração:** O LLM usa o contexto para gerar ameaças mais precisas e fundamentadas

### Configurações RAG:

| Parâmetro | Default | Descrição |
|---|---|---|
| `RAG_TOP_K` | 6 | Número máximo de snippets recuperados por componente |
| `rag_min_score` | 0.25 | Score mínimo de similaridade (cosseno normalizado) |

### Estrutura da KB:

```
60 entradas = 14 classes × ~4-5 categorias STRIDE relevantes por classe

Cada entrada contém:
├── id: identificador único (ex: "stride_database_tampering")
├── component: nome da classe (ex: "Database")
├── category: categoria STRIDE (ex: "Tampering")
└── text: descrição técnica com ameaças, vetores de ataque e mitigações
```

---

## 11. Variáveis de Ambiente

Todas as variáveis são configuradas no arquivo `.env` (copie de `.env.example`):

| Variável | Default | Obrigatória | Descrição |
|---|---|---|---|
| `ENV` | development | Não | Ambiente (development/production) |
| `HOST` | 0.0.0.0 | Não | Host do servidor |
| `PORT` | 8000 | Não | Porta do servidor |
| `MODEL_PATH` | ./models/best.pt | Não | Caminho para os pesos YOLO |
| `DATA_YAML_PATH` | ./data/data.yaml | Não | Configuração do dataset YOLO |
| `KB_PATH` | data/kb.json | Não | Caminho para a base de conhecimento |
| `OPENAI_API_KEY` | — | **Sim** | Chave da API OpenAI |
| `OPENAI_MODEL` | gpt-4.1-mini | Não | Modelo LLM |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-small | Não | Modelo de embeddings |
| `RAG_ENABLED` | true | Não | Ativar/desativar RAG |
| `RAG_TOP_K` | 6 | Não | Quantidade de snippets RAG |
| `CONFIDENCE_THRESHOLD` | 0.35 | Não | Threshold de confiança do YOLO |
| `IOU_DUPLICATE_THRESHOLD` | 0.5 | Não | Threshold IoU para NMS |
| `MAX_IMAGE_MB` | 10 | Não | Tamanho máximo de upload (MB) |

---

## 12. Estrutura de Diretórios

```
fiap-hackathon-threat-modeling/
│
├── backend/                    # Código-fonte do backend
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + rotas
│   ├── detector.py             # YOLO wrapper
│   ├── stride_engine.py        # Motor STRIDE (LLM + RAG)
│   ├── kb.py                   # Loader da base de conhecimento
│   ├── report_generator.py     # Gerador de PDF (ReportLab)
│   ├── schemas.py              # Modelos Pydantic v2
│   └── settings.py             # Configurações (.env)
│
├── data/                       # Dados e configuração de dataset
│   ├── data.yaml               # Config YOLO (14 classes + paths)
│   ├── kb.json                 # Base de conhecimento (60 entradas)
│   ├── raw_icons/              # Ícones para dataset (opcional)
│   ├── images/                 # Dataset gerado
│   │   ├── all/                # Todas as imagens
│   │   ├── train/              # Conjunto de treino (80%)
│   │   └── val/                # Conjunto de validação (20%)
│   └── labels/                 # Labels YOLO correspondentes
│       ├── all/
│       ├── train/
│       └── val/
│
├── docs/                       # Documentação
│   ├── MANUAL.md               # Este manual
│   ├── STATUS.md               # Status do projeto
│   ├── README.md               # README geral
│   └── architecture.md         # Diagrama de arquitetura
│
├── models/                     # Modelos YOLO
│   ├── yolov8n.pt              # Modelo base (pré-treinado COCO)
│   └── best.pt                 # Modelo treinado (gerado)
│
├── outputs/                    # PDFs gerados
│   └── report_*.pdf
│
├── scripts/                    # Scripts ML
│   ├── generate_dataset.py     # Gerador de dataset sintético
│   ├── split_dataset.py        # Divisão train/val
│   ├── train_yolo.py           # Treinamento YOLO
│   └── evaluate.py             # Avaliação de métricas
│
├── static/                     # Arquivos estáticos (CSS, JS)
│   ├── style.css
│   └── app.js
│
├── templates/                  # Templates HTML (Jinja2)
│   ├── index.html              # Página de upload
│   └── result.html             # Página de resultado
│
├── runs/                       # Logs de treinamento YOLO (gerado)
├── .env                        # Variáveis de ambiente (NÃO commitar)
├── .env.example                # Template das variáveis
├── .gitignore
└── requirements.txt            # Dependências Python
```

---

## 13. Solução de Problemas

### Erro: `OPENAI_API_KEY não configurada`
**Causa:** Arquivo `.env` sem chave ou chave vazia.  
**Solução:** Adicione `OPENAI_API_KEY=sk-...` no arquivo `.env`.

### Erro: `YOLO weights not found`
**Causa:** Modelo `best.pt` não existe em `models/`.  
**Solução:** Execute o pipeline de treinamento (seção 5) ou aponte `MODEL_PATH` para outro `.pt`.

### Erro: `YOLO model produced unknown class label`
**Causa:** Modelo carregado é o `yolov8n.pt` base (COCO, 80 classes) ao invés do modelo customizado.  
**Solução:** Configure `MODEL_PATH=./models/best.pt` no `.env`.

### Erro: `Dataset images not found`
**Causa:** `data.yaml` com caminhos incorretos ou dataset não gerado.  
**Solução:** Execute `generate_dataset.py` e `split_dataset.py` antes de treinar.

### Erro: `Image too large`
**Causa:** Upload excede `MAX_IMAGE_MB` (default: 10 MB).  
**Solução:** Redimensione a imagem ou aumente `MAX_IMAGE_MB` no `.env`.

### Erro: `LLM returned empty content` ou `429 Too Many Requests`
**Causa:** Limite de rate da OpenAI atingido ou problema de conectividade.  
**Solução:** Aguarde alguns segundos e tente novamente. Verifique seu plano da OpenAI.

### Servidor não responde
**Causa:** Porta 8000 em uso ou venv não ativado.  
**Solução:** Verifique com `lsof -i :8000` (Linux) ou `netstat -ano | findstr 8000` (Windows). Certifique-se de ativar o ambiente virtual antes de iniciar.

### Detecções com baixa confiança
**Causa:** Modelo treinado com poucas épocas ou dataset insuficiente.  
**Solução:** Aumente `--num-images` para 500+ e `--epochs` para 40+. Adicione ícones reais em `data/raw_icons/`.

---

## 14. Glossário

| Termo | Definição |
|---|---|
| **STRIDE** | Metodologia de modelagem de ameaças: Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege |
| **YOLO** | You Only Look Once — arquitetura de detecção de objetos em tempo real |
| **YOLOv8n** | Versão nano (leve) do YOLO v8, ideal para inferência em CPU |
| **RAG** | Retrieval-Augmented Generation — técnica que combina busca de informações com geração por LLM |
| **mAP@0.5** | Mean Average Precision com IoU threshold de 50% — métrica principal de detecção |
| **NMS** | Non-Maximum Suppression — remove detecções duplicadas/sobrepostas |
| **IoU** | Intersection over Union — mede sobreposição entre bounding boxes |
| **Bounding Box** | Retângulo que delimita um objeto detectado na imagem |
| **Transfer Learning** | Reutilização de modelo pré-treinado para nova tarefa |
| **Embeddings** | Representações vetoriais de texto para busca semântica |
| **Similaridade de Cosseno** | Métrica de semelhança entre vetores (0 = diferente, 1 = idêntico) |
| **LLM** | Large Language Model — modelo de linguagem de grande porte |
| **MVP** | Minimum Viable Product — versão mínima funcional do produto |
| **Pydantic** | Biblioteca Python para validação de dados via schemas |
| **FastAPI** | Framework web moderno e rápido para Python |
| **ReportLab** | Biblioteca para geração programática de PDFs |
| **Augmentation** | Técnica de aumento de dados (rotação, escala, brilho, etc.) |

---

> **Documento gerado para o Hackathon FIAP — Software Security 2026**
