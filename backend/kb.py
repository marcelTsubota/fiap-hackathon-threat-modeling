"""
backend/kb.py

Local knowledge‑base loader / retriever with in‑memory embedding cache.

Designed for the Hackathon FIAP Threat Modeling MVP.  The KB is a simple
JSON file; the module can be used standalone or by the STRIDE engine.

Responsibilities
---------------
* load/validate KB JSON (list or {items:[…]})
* normalise each entry to have a “key” and a “text” field
* honour pre‑computed embeddings (list of floats)
* lazily compute embeddings via OpenAI (text-embedding-3-small)
* cache all vectors for the lifetime of the process
* retrieve top‑k hits by cosine similarity with a minimum score threshold

Errors
------
KBLoadError      – invalid or unreadable KB file  
LLMRequiredError – embeddings requested but OPENAI_API_KEY missing
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI

from backend.schemas import KBHit
from backend.settings import settings


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KBLoadError(RuntimeError):
    """Raised when the KB file exists but cannot be parsed."""


class LLMRequiredError(RuntimeError):
    """Raised when embeddings are needed but no OpenAI key is available."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_vec(v: Sequence[float]) -> np.ndarray:
    """Return a float32 vector normalised to unit length (handles zero vector)."""
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm != 0.0 else arr


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class LocalKB:
    """
    Minimal local KB implementation.

    The underlying JSON may be either a list of objects or an object containing
    an ``items`` list.  Each item is normalised to a dict with:

        { "key": ..., "text": ..., **original_fields }

    If an item already provides ``embedding`` (list of numbers) it is used as‑is
    (after normalisation); otherwise the embedding is computed on demand using
    the OpenAI embeddings API and cached in memory.

    Retrieval is by cosine similarity; scores are linearly mapped to 0..1 and
    thresholded against ``min_score``.
    """

    def __init__(
        self,
        kb_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        min_score: float = 0.25,
    ) -> None:
        # path may be overridden by settings.KB_PATH
        self.kb_path: str = kb_path or getattr(settings, "KB_PATH", "data/kb.json")
        self.embedding_model: str = (
            embedding_model
            or getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
        self.min_score: float = min_score

        self._loaded: bool = False
        self._items: List[Dict[str, Any]] = []
        self._embeddings: List[np.ndarray] = []
        self._client: Optional[OpenAI] = None
        self._api_key: Optional[str] = (
            getattr(settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
        )

    # ------------------------------------------------------------------
    # Loading / normalisation
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load and normalise the KB file.  Idempotent."""
        if self._loaded:
            return

        path = Path(self.kb_path)
        if not path.exists():
            # KB is optional; an empty knowledge base is valid.
            self._loaded = True
            self._items = []
            self._embeddings = []
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise KBLoadError(f"Failed to load KB JSON at {path}") from exc

        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict) and isinstance(raw.get("items"), list):
            items = raw["items"]
        else:
            raise KBLoadError("KB JSON must be a list or an object with an 'items' list.")

        norm_items: List[Dict[str, Any]] = []
        norm_emb: List[np.ndarray] = []

        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                # skip malformed entries silently
                continue

            key = str(it.get("id") or it.get("key") or f"kb_{idx}")

            text = it.get("text")
            if not isinstance(text, str) or not text.strip():
                # fallback: stringify the whole object
                text = json.dumps(it, ensure_ascii=False)

            entry = {"key": key, "text": text, **it}

            emb = it.get("embedding")
            if (
                isinstance(emb, list)
                and emb
                and all(isinstance(x, (int, float)) for x in emb)
            ):
                norm_emb.append(_normalize_vec(emb))
            else:
                norm_emb.append(np.array([], dtype=np.float32))  # placeholder

            norm_items.append(entry)

        self._loaded = True
        self._items = norm_items
        self._embeddings = norm_emb

    @property
    def items(self) -> List[Dict[str, Any]]:
        self.load()
        return self._items

    @property
    def embeddings(self) -> List[np.ndarray]:
        self.load()
        return self._embeddings

    def _ensure_client(self) -> None:
        """Initialise OpenAI client, raising if API key is missing."""
        if self._client is not None:
            return
        if not self._api_key:
            raise LLMRequiredError("OPENAI_API_KEY is required for embeddings.")
        self._client = OpenAI(api_key=self._api_key)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Request an embedding for the given text and normalise it."""
        self._ensure_client()
        try:
            resp = self._client.embeddings.create(
                model=self.embedding_model, input=text
            )
            vec = resp.data[0].embedding  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(f"embedding request failed: {exc}") from exc

        return _normalize_vec(np.array(vec, dtype=np.float32))

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> List[KBHit]:
        """
        Return up to ``top_k`` KBHit objects whose text is most similar to
        ``query``.  Only hits with mapped similarity >= ``min_score`` are
        returned.  If no embeddings exist yet they are computed lazily.
        """
        if not query:
            return []

        q_vec = self._get_embedding(query)

        # compute missing item embeddings
        for idx, vec in enumerate(self.embeddings):
            if vec.size == 0:
                text = self.items[idx]["text"]
                self._embeddings[idx] = self._get_embedding(text)

        scored: List[Tuple[float, int]] = []
        for idx, vec in enumerate(self.embeddings):
            if vec.size == 0:
                continue
            cos = float(np.dot(q_vec, vec))  # -1..1
            score01 = max(0.0, min((cos + 1.0) / 2.0, 1.0))
            if score01 >= self.min_score:
                scored.append((score01, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        hits: List[KBHit] = []
        filename = Path(self.kb_path).name
        for score, idx in scored[:top_k]:
            item = self.items[idx]
            snippet: Optional[str] = None
            txt = item.get("text")
            if isinstance(txt, str):
                snippet = txt[:200]
            hits.append(
                KBHit(source=filename, key=item.get("key"), score=score, snippet=snippet)
            )

        return hits
