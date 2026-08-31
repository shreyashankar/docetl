"""The `StippleVerifyOperation` — optional document-authenticity verification.

Inspects the source document referenced by each row (typically a PDF produced
by a upstream `map`/`code_map` step, or any local/HTTP path present in a
configurable document key) via the Stipple REST API and attaches the result —
a risk band, inspection quality, recommended action, and a re-verifiable
warrant id — to the row.

Why: docetl pipelines trust their input documents. When the source corpus
may contain tampered or synthetic documents, this operation adds a
trust signal that downstream filter/map steps can act on:

    - name: verify_sources
      type: stipple_verify
      doc_path_key: pdf_path        # row key holding the document path/URL
      output_key: document_trust    # key the warrant is written to
      # fail_on: ["high"]           # optional: drop documents above a band

Free anonymous tier (no signup); set the STIPPLE_API_KEY environment
variable for your own metering. All failures are best-effort: the operation
never raises on network/API errors — it records `document_trust.error`
instead, and honors fail_on only when a verdict is actually available.
"""

import os
import tempfile
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from pydantic import Field, model_validator

from docetl.operations.base import BaseOperation, Cardinality
from docetl.operations.utils import RichLoopBar

DEFAULT_STIPPLE_BASE_URL = "https://www.stipple.sh"
DEFAULT_TIMEOUT = 300  # seconds
VALID_BANDS = ("low", "medium", "high")


class StippleVerifyOperation(BaseOperation):
    class schema(BaseOperation.schema):
        type: str = "stipple_verify"
        doc_path_key: str = Field(
            description="Row key holding the path/URL of the source document"
        )
        output_key: str = Field(
            default="document_trust",
            description="Key the verification warrant is written to",
        )
        base_url: str = Field(default=DEFAULT_STIPPLE_BASE_URL)
        fail_on: list[str] | None = Field(
            default=None,
            description="Risk bands (e.g. ['high']) whose documents are dropped from the output",
        )
        verify_ai_text: bool = Field(
            default=True,
            description="Also run AI-written-text detection on the document",
        )
        timeout: float = Field(default=300.0, gt=0)
        max_workers: int = Field(default=4, gt=0)

        @model_validator(mode="after")
        def validate_bands(self):
            if self.fail_on is not None:
                bad = [b for b in self.fail_on if b not in VALID_BANDS]
                if bad:
                    raise ValueError(
                        f"fail_on contains invalid risk bands: {bad}. "
                        f"Valid bands: {VALID_BANDS}"
                    )
            return self

    @property
    def fail_on(self) -> list[str] | None:
        return self.config.get("fail_on")

    @classmethod
    def cardinality(cls, config: dict[str, Any]) -> Cardinality:
        return Cardinality.ONE_TO_ONE

    def syntax_check(self) -> None:
        config = self.schema(**self.config)
        _ = config  # validation happens in the pydantic schema

    # ── Stipple REST helpers (best-effort; stdlib + requests only) ─────

    def _headers(self) -> dict:
        headers = {
            "User-Agent": "docetl-stipple-verify/1.0",
            "Accept": "application/json",
        }
        api_key = os.getenv("STIPPLE_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        return headers

    def _post_file(self, endpoint: str, source: str) -> dict | None:
        """POST a local path or HTTP(S) URL as multipart to a Stipple endpoint."""
        try:
            source_path = self._resolve_local(source)
        except Exception:
            return None
        boundary = "----docetl-stipple" + uuid.uuid4().hex
        try:
            if source_path is not None:
                with open(source_path, "rb") as f:
                    content = f.read()
                filename = Path(source_path).name
            else:
                # Remote document: fetch it, then upload (intake is upload-based)
                with urllib.request.urlopen(source, timeout=self.config.get("timeout", 300.0)) as r:
                    content = r.read()
                filename = source.rsplit("/", 1)[-1] or "document"
            body = b"".join(
                [
                    (
                        f"--{boundary}\r\n"
                        f'Content-Disposition: form-data; name="file"; '
                        f'filename="{filename}"\r\n'
                        "Content-Type: application/octet-stream\r\n\r\n"
                    ).encode(),
                    content,
                    b"\r\n",
                    f"--{boundary}--\r\n".encode(),
                ]
            )
            resp = requests.post(
                self.config.get("base_url", DEFAULT_STIPPLE_BASE_URL) + endpoint,
                data=body,
                headers={
                    **self._headers(),
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                timeout=self.config.get("timeout", 300.0),
            )
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    def _resolve_local(self, source: str) -> str | None:
        """Return a local file path for the source, downloading if needed."""
        if source.startswith(("http://", "https://")):
            with urllib.request.urlopen(source, timeout=self.config.get("timeout", 300.0)) as r:
                data = r.read()
            suffix = Path(source.rsplit("/", 1)[-1].split("?")[0]).suffix or ".pdf"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data)
            tmp.close()
            return tmp.name
        p = Path(source)
        if p.is_file():
            return str(p)
        return None

    def _verify(self, doc: dict) -> dict:
        """Verify one document. Always returns a dict; errors are recorded."""
        row_key = self.schema(**self.config).doc_path_key
        output_key = self.schema(**self.config).output_key
        source = doc.get(row_key)
        if not source:
            return {output_key: {"error": f"no value for '{row_key}'"}}

        result: dict[str, Any] = {}
        warrant = self._post_file("/v1/warrants", source)
        if warrant:
            result = {
                "warrant_id": warrant.get("warrant_id"),
                "risk_band": warrant.get("risk_band"),
                "risk_score": warrant.get("risk_score"),
                "inspection_quality": warrant.get("inspection_quality"),
                "recommended_action": warrant.get("recommended_action"),
                "summary": warrant.get("summary"),
            }
        else:
            result = {"error": "authenticity inspection unavailable"}
        if self.schema(**self.config).verify_ai_text:
            ai = self._post_file("/v1/detect-ai-text", source)
            if ai:
                if ai.get("applicable") is False:
                    result["ai_text"] = {"applicable": False}
                else:
                    result["ai_text"] = {
                        "applicable": True,
                        "probability": ai.get("probability"),
                        "lean": ai.get("lean"),
                        "tells": ai.get("tells"),
                    }
        return {output_key: result}

    # ── execution ─────────────────────────────────────────────────────

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        config = self.schema(**self.config)
        results = []
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [executor.submit(self._verify, doc) for doc in input_data]
            pbar = RichLoopBar(
                range(len(futures)),
                desc=f"Processing {self.config['name']} (stipple_verify)",
                console=self.console,
            )
            for i in pbar:
                doc = dict(input_data[i])
                verdict = futures[i].result()
                merged = {**doc, **verdict}
                # Optional policy enforcement: drop documents whose risk band
                # is in fail_on. Documents without a verdict are kept.
                if self.fail_on:
                    verdict = merged.get(config.output_key, {})
                    band = verdict.get("risk_band")
                    if band in self.fail_on:
                        continue
                results.append(merged)
        return results, 0.0
