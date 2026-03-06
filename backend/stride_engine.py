# backend/stride_engine.py
"""
STRIDE Engine (LLM + RAG) for the FIAP Hackathon MVP.

Responsibilities:
- Input: DiagramDetections (from YOLO detector)
- Output: StrideResult (list of Threat items)
- LLM is mandatory by default (fails hard if OPENAI_API_KEY missing)
- RAG enabled: local KB JSON + embeddings (text-embedding-3-small)

Non-responsibilities:
- YOLO inference
- FastAPI routes / request handling
- PDF generation
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI

from backend.schemas import (
    ComponentClass,
    DiagramDetections,
    KBHit,
    LLMUsage,
    Severity,
    StrideCategory,
    StrideResult,
    Threat,
)
from backend.settings import settings


# -----------------------------
# Exceptions
# -----------------------------

class LLMRequiredError(RuntimeError):
    """Raised when LLM is required but OPENAI_API_KEY is missing."""


class LLMResponseError(RuntimeError):
    """Raised when the LLM response is invalid or cannot be parsed/validated."""


class KBLoadError(RuntimeError):
    """Raised when KB file exists but cannot be loaded/parsed."""


# -----------------------------
# STRIDE mapping
# -----------------------------
# Practical mapping (conservative defaults). LLM can still refine.
_COMPONENT_TO_STRIDE: Dict[ComponentClass, List[StrideCategory]] = {
    ComponentClass.USER: [
        StrideCategory.SPOOFING,
        StrideCategory.REPUDIATION,
        StrideCategory.INFORMATION_DISCLOSURE,
    ],
    ComponentClass.EXTERNAL_SYSTEM: [
        StrideCategory.SPOOFING,
        StrideCategory.TAMPERING,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
    ],
    ComponentClass.API_GATEWAY: [
        StrideCategory.SPOOFING,
        StrideCategory.TAMPERING,
        StrideCategory.REPUDIATION,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
        StrideCategory.ELEVATION_OF_PRIVILEGE,
    ],
    ComponentClass.LOAD_BALANCER: [
        StrideCategory.TAMPERING,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
    ],
    ComponentClass.WEB_APPLICATION: [
        StrideCategory.SPOOFING,
        StrideCategory.TAMPERING,
        StrideCategory.REPUDIATION,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
        StrideCategory.ELEVATION_OF_PRIVILEGE,
    ],
    ComponentClass.MOBILE_APPLICATION: [
        StrideCategory.SPOOFING,
        StrideCategory.TAMPERING,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.ELEVATION_OF_PRIVILEGE,
    ],
    ComponentClass.APPLICATION_SERVER: [
        StrideCategory.TAMPERING,
        StrideCategory.REPUDIATION,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
        StrideCategory.ELEVATION_OF_PRIVILEGE,
    ],
    ComponentClass.SERVERLESS_FUNCTION: [
        StrideCategory.TAMPERING,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
        StrideCategory.ELEVATION_OF_PRIVILEGE,
    ],
    ComponentClass.DATABASE: [
        StrideCategory.TAMPERING,
        StrideCategory.REPUDIATION,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
        StrideCategory.ELEVATION_OF_PRIVILEGE,
    ],
    ComponentClass.CACHE: [
        StrideCategory.TAMPERING,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
    ],
    ComponentClass.MESSAGE_QUEUE: [
        StrideCategory.TAMPERING,
        StrideCategory.REPUDIATION,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
    ],
    ComponentClass.FILE_STORAGE: [
        StrideCategory.TAMPERING,
        StrideCategory.REPUDIATION,
        StrideCategory.INFORMATION_DISCLOSURE,
    ],
    ComponentClass.AUTHENTICATION_SERVICE: [
        StrideCategory.SPOOFING,
        StrideCategory.TAMPERING,
        StrideCategory.REPUDIATION,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
        StrideCategory.ELEVATION_OF_PRIVILEGE,
    ],
    ComponentClass.SECURITY_SERVICE: [
        StrideCategory.TAMPERING,
        StrideCategory.INFORMATION_DISCLOSURE,
        StrideCategory.DENIAL_OF_SERVICE,
        StrideCategory.ELEVATION_OF_PRIVILEGE,
    ],
}


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class StrideEngineConfig:
    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    # RAG / KB
    kb_path: str = "data/kb.json"
    rag_top_k: int = 5
    rag_min_score: float = 0.25  # cosine similarity threshold (0..1)
    # Output shaping
    threats_per_component: int = 3


# -----------------------------
# KB (local JSON) + embeddings
# -----------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize_vec(v: Sequence[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n == 0:
        return arr
    return arr / n


class _LocalKB:
    """
    Minimal local KB loader.
    Expected file format (flexible):
    - list of objects OR dict with "items"
    Each item may contain:
      - "id" or "key"
      - "text" (recommended) OR any fields (we will stringify)
      - optionally "embedding" (list[float]) to skip embedding at runtime
    """

    def __init__(self, kb_path: str) -> None:
        self.kb_path = kb_path
        self._loaded: bool = False
        self._items: List[Dict[str, Any]] = []
        self._embeddings: List[np.ndarray] = []

    def load(self) -> None:
        if self._loaded:
            return

        path = Path(self.kb_path)
        if not path.exists():
            # KB optional at runtime; RAG will degrade gracefully.
            self._loaded = True
            self._items = []
            self._embeddings = []
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise KBLoadError(f"Failed to load KB JSON at {str(path)}") from exc

        items: Any
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict) and isinstance(raw.get("items"), list):
            items = raw["items"]
        else:
            raise KBLoadError("KB JSON must be a list or an object with an 'items' list.")

        norm_items: List[Dict[str, Any]] = []
        norm_embeddings: List[np.ndarray] = []

        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            key = str(it.get("id") or it.get("key") or f"kb_{idx}")
            # Prefer an explicit text field, otherwise stringify the whole item
            text = it.get("text")
            if not isinstance(text, str) or not text.strip():
                text = json.dumps(it, ensure_ascii=False)
            it2 = {"key": key, "text": text, **it}

            emb = it.get("embedding")
            if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
                norm_embeddings.append(_normalize_vec(emb))
            else:
                norm_embeddings.append(np.array([], dtype=np.float32))  # to be filled later

            norm_items.append(it2)

        self._loaded = True
        self._items = norm_items
        self._embeddings = norm_embeddings

    @property
    def items(self) -> List[Dict[str, Any]]:
        self.load()
        return self._items

    @property
    def embeddings(self) -> List[np.ndarray]:
        self.load()
        return self._embeddings

    def set_embedding(self, idx: int, emb: Sequence[float]) -> None:
        self._embeddings[idx] = _normalize_vec(emb)


# -----------------------------
# Engine
# -----------------------------

class StrideEngine:
    def __init__(self, config: Optional[StrideEngineConfig] = None) -> None:
        self.config = config or StrideEngineConfig(
            llm_model=getattr(settings, "OPENAI_LLM_MODEL", "gpt-4.1-mini"),
            embedding_model=getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            kb_path=getattr(settings, "KB_PATH", "data/kb.json"),
        )

        api_key = getattr(settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMRequiredError(
                "OPENAI_API_KEY is required by project rules (LLM obrigatório)."
            )

        self.client = OpenAI(api_key=api_key)
        self.kb = _LocalKB(self.config.kb_path)

    # -------------------------
    # Public API
    # -------------------------

    def analyze(self, diagram: DiagramDetections) -> StrideResult:
        """
        Main entry:
        - Builds a structured prompt using detections + STRIDE mapping
        - Retrieves KB hits via embeddings (RAG)
        - Calls LLM and validates Threat outputs via Pydantic schema
        """
        # RAG context (best effort; empty if KB missing)
        rag_context = self._build_rag_context(diagram)

        prompt = self._build_prompt(diagram=diagram, rag_context=rag_context)
        raw, usage = self._call_llm_json(prompt)

        threats = self._parse_llm_output(raw, diagram)
        # Attach basic usage telemetry into Threat.confidence only; detailed usage can be used later in report bundle.
        # Here, keep schema pure and deterministic.
        _ = usage  # retained for future enrichment bundle; not stored in StrideResult per current schema.

        return StrideResult(threats=threats)

    # -------------------------
    # Prompting
    # -------------------------

    def _build_prompt(self, diagram: DiagramDetections, rag_context: str) -> str:
        """
        Produce a single JSON-only instruction prompt.
        We keep it concise but strict: LLM must output ONLY valid JSON.
        """
        components_payload: List[Dict[str, Any]] = []
        for d in diagram.detections:
            stride_targets = [c.value for c in _COMPONENT_TO_STRIDE.get(d.label, list(StrideCategory))]
            components_payload.append(
                {
                    "component_id": d.id,
                    "component_label": d.label.value,
                    "confidence": d.confidence,
                    "bbox_xyxy": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                    "stride_categories_allowed": stride_targets,
                }
            )

        schema_hint = {
            "threats": [
                {
                    "component_id": "cmp_...",
                    "component_label": "Database",
                    "category": "Tampering",
                    "title": "Short threat name",
                    "description": "What could happen and how (10+ chars).",
                    "impact": "Business/security impact.",
                    "mitigations": ["Mitigation 1", "Mitigation 2"],
                    "severity": "High",
                    "confidence": 0.0,
                }
            ]
        }

        return (
            "You are a senior application security engineer. "
            "Generate STRIDE threat modeling results for the detected architecture components.\n\n"
            "STRICT OUTPUT RULES:\n"
            "1) Output MUST be valid JSON only (no markdown, no extra text).\n"
            "2) Output JSON MUST match this top-level shape: {\"threats\": [ ... ]}.\n"
            "3) Use only the provided components; do not invent components.\n"
            "4) For each component, produce exactly "
            f"{self.config.threats_per_component} threats.\n"
            "5) 'component_label' must be EXACTLY one of the fixed class names provided.\n"
            "6) 'category' must be EXACTLY one of: "
            f"{[c.value for c in StrideCategory]}.\n"
            "7) 'severity' must be EXACTLY one of: "
            f"{[s.value for s in Severity]}.\n"
            "8) 'confidence' must be a float 0..1.\n"
            "9) Mitigations must be actionable and specific.\n\n"
            "REFERENCE SCHEMA EXAMPLE (do not copy blindly):\n"
            f"{json.dumps(schema_hint, ensure_ascii=False)}\n\n"
            "DETECTED COMPONENTS:\n"
            f"{json.dumps(components_payload, ensure_ascii=False)}\n\n"
            "RAG CONTEXT (grounding snippets; may be empty):\n"
            f"{rag_context}\n"
        )

    # -------------------------
    # LLM calls (JSON-only)
    # -------------------------

    def _call_llm_json(self, prompt: str) -> Tuple[Dict[str, Any], Optional[LLMUsage]]:
        """
        Calls OpenAI Chat Completions API and enforces JSON-only output via prompting + strict parsing.

        Note:
        - Some SDK builds do not support `response_format` on Chat Completions.
        - We enforce JSON by instruction and then parse strictly.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior application security engineer. "
                            "Return ONLY valid JSON. Do not include markdown, code fences, or any extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
        except Exception as exc:
            raise LLMResponseError(f"LLM call failed: {exc}") from exc

        try:
            raw_text = resp.choices[0].message.content
        except Exception as exc:
            raise LLMResponseError("Unexpected LLM response shape.") from exc

        if not raw_text or not raw_text.strip():
            raise LLMResponseError("LLM returned empty content.")

        text = raw_text.strip()

        # Strict JSON parse first
        try:
            data = json.loads(text)
        except Exception:
            # Fallback: try to extract the first JSON object region
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise LLMResponseError("LLM did not return valid JSON (no JSON object found).")
            candidate = text[start : end + 1]
            try:
                data = json.loads(candidate)
            except Exception as exc:
                raise LLMResponseError("LLM did not return valid JSON.") from exc

        usage_obj = getattr(resp, "usage", None)
        usage: Optional[LLMUsage] = None
        if usage_obj:
            usage = LLMUsage(
                model=self.config.llm_model,
                prompt_tokens=getattr(usage_obj, "prompt_tokens", None),
                completion_tokens=getattr(usage_obj, "completion_tokens", None),
                total_tokens=getattr(usage_obj, "total_tokens", None),
            )

        return data, usage
    # -------------------------
    # Parsing + validation
    # -------------------------

    def _parse_llm_output(self, data: Dict[str, Any], diagram: DiagramDetections) -> List[Threat]:
        """
        Validate JSON structure and convert to Threat objects.
        Threat IDs are generated here to be stable inside this run.
        """
        if not isinstance(data, dict) or "threats" not in data:
            raise LLMResponseError("LLM JSON must be an object with top-level key 'threats'.")

        threats_raw = data["threats"]
        if not isinstance(threats_raw, list):
            raise LLMResponseError("'threats' must be a list.")

        # Build a map to validate component_id references
        det_map: Dict[str, ComponentClass] = {d.id: d.label for d in diagram.detections}

        out: List[Threat] = []
        for item in threats_raw:
            if not isinstance(item, dict):
                continue

            component_id = str(item.get("component_id", "")).strip()
            if component_id not in det_map:
                raise LLMResponseError(f"Threat references unknown component_id: {component_id}")

            # Enforce fixed label from detector (ignore LLM hallucinations)
            component_label = det_map[component_id]

            try:
                category = StrideCategory(str(item.get("category", "")).strip())
                severity = Severity(str(item.get("severity", "")).strip())
            except ValueError as exc:
                raise LLMResponseError(f"Invalid category/severity in LLM output: {item}") from exc

            mitigations = item.get("mitigations", [])
            if mitigations is None:
                mitigations = []
            if not isinstance(mitigations, list):
                mitigations = [str(mitigations)]

            threat = Threat(
                threat_id=self._make_threat_id(),
                component_id=component_id,
                component_label=component_label,
                category=category,
                title=str(item.get("title", "")).strip(),
                description=str(item.get("description", "")).strip(),
                impact=str(item.get("impact", "")).strip(),
                mitigations=[str(m).strip() for m in mitigations if str(m).strip()],
                severity=severity,
                confidence=float(item.get("confidence", 0.75)),
            )
            out.append(threat)

        # Enforce exactly N threats per component.
        # If the LLM returns more than expected, keep the top-N by severity.
        # If fewer, accept what we got (better partial results than a crash).
        expected = self.config.threats_per_component
        _SEV_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

        grouped: Dict[str, List[Threat]] = {}
        for t in out:
            grouped.setdefault(t.component_id, []).append(t)

        trimmed: List[Threat] = []
        for comp_id, threats in grouped.items():
            if len(threats) > expected:
                threats.sort(
                    key=lambda x: _SEV_RANK.get(x.severity.value, 0), reverse=True
                )
                threats = threats[:expected]
            trimmed.extend(threats)

        return trimmed

    @staticmethod
    def _make_threat_id() -> str:
        return f"thr_{uuid.uuid4().hex[:12]}"

    # -------------------------
    # RAG (embeddings + retrieval)
    # -------------------------

    def _embed_text(self, text: str) -> List[float]:
        """
        Embed text using OpenAI embeddings model.
        """
        try:
            resp = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=text,
            )
        except Exception as exc:
            raise LLMResponseError(f"Embedding call failed: {exc}") from exc

        emb = resp.data[0].embedding
        if not isinstance(emb, list) or not emb:
            raise LLMResponseError("Embedding API returned an empty vector.")
        return emb

    def _ensure_kb_embeddings(self) -> None:
        """
        Ensures each KB item has an embedding (computed lazily on first use).
        """
        items = self.kb.items
        embs = self.kb.embeddings

        if not items:
            return

        for i, (it, emb) in enumerate(zip(items, embs)):
            if emb.size != 0:
                continue
            text = str(it.get("text", "")).strip()
            if not text:
                continue
            vec = self._embed_text(text)
            self.kb.set_embedding(i, vec)

    def _retrieve_kb(self, query: str, top_k: int) -> List[KBHit]:
        """
        Retrieves top_k KB items by cosine similarity.
        Returns KBHit objects with score normalized to 0..1 (cosine is -1..1, here clipped).
        """
        self.kb.load()
        if not self.kb.items:
            return []

        self._ensure_kb_embeddings()

        qv = _normalize_vec(self._embed_text(query))
        scored: List[Tuple[int, float]] = []
        for idx, ev in enumerate(self.kb.embeddings):
            if ev.size == 0:
                continue
            sim = _cosine_sim(qv, ev)  # -1..1
            # Clip to [0,1] for our KBHit schema
            score01 = max(0.0, min((sim + 1.0) / 2.0, 1.0))
            scored.append((idx, score01))

        scored.sort(key=lambda x: x[1], reverse=True)
        hits: List[KBHit] = []
        for idx, score in scored[:top_k]:
            if score < self.config.rag_min_score:
                continue
            it = self.kb.items[idx]
            hits.append(
                KBHit(
                    source=str(Path(self.config.kb_path).name),
                    key=str(it.get("key", f"kb_{idx}")),
                    score=score,
                    snippet=str(it.get("text", ""))[:500] or None,
                )
            )
        return hits

    def _build_rag_context(self, diagram: DiagramDetections) -> str:
        """
        Builds a grounding string from KB hits per component.
        This is intentionally plain text to keep prompting simple and robust.
        """
        if not diagram.detections:
            return ""

        blocks: List[str] = []
        for det in diagram.detections:
            stride_targets = _COMPONENT_TO_STRIDE.get(det.label, list(StrideCategory))
            query = f"{det.label.value} threats mitigations STRIDE: {[s.value for s in stride_targets]}"
            hits = self._retrieve_kb(query=query, top_k=self.config.rag_top_k)
            if not hits:
                continue

            blocks.append(f"Component {det.id} ({det.label.value}) KB hits:")
            for h in hits:
                blocks.append(f"- [{h.key}] score={h.score:.2f} :: {h.snippet or ''}")
            blocks.append("")  # spacing

        return "\n".join(blocks).strip()