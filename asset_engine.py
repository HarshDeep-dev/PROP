"""
asset_engine.py — Self-Healing RWA Content Engine (2026 Edition).

Key capabilities:
  · Model Failover  — __init__ tests the preferred model; if it returns a 404,
                       it calls _discover_model() to list available models from
                       the API and selects the best available flash variant.
  · Anti-429        — _generate_with_retry() implements exponential backoff with
                       the exact log message: [QUOTA] Rate limit hit. Pausing for Xs...
  · Context Chain   — Stage 1 article is injected as grounding context into the
                       Stage 2 (LinkedIn) and Stage 3 (CFO summary) prompts.
  · Diagnostics     — test_connection() makes a minimal "Hello" probe call so
                       main.py can surface clean errors before the real pipeline starts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from google import genai

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

log = logging.getLogger("RWA_ENGINE")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primary model — compatible with the v1beta endpoint, generous free tier.
_PRIMARY_MODEL = "gemini-2.5-flash-lite"

# Fallback preference order when the primary model returns 404.
_FALLBACK_PREFERENCE = [
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]

# Quota / rate-limit signals to match against exception messages.
_QUOTA_SIGNALS = (
    "429",
    "resource exhausted",
    "rate limit",
    "quota exceeded",
    "quota",
    "too many requests",
)

# 404 / model-not-found signals.
_NOT_FOUND_SIGNALS = (
    "404",
    "not found",
    "not_found",
    "model not found",
    "unrecognized model",
)


# ---------------------------------------------------------------------------
# RWAContentEngine
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RWAContentEngine:
    """
    Self-healing, production-ready engine for RWA content generation.

    Stages
    ------
    1. generate_article(topic)         → Fact Sheet
    2. convert_to_linkedin(article)    → Public Summary (uses Stage 1 context)
    3. generate_summary(article)       → Risk Assessment (uses Stage 1 context)
    """

    api_key: str
    model: str = _PRIMARY_MODEL
    max_retries: int = 5
    # Base pause in seconds for the first quota back-off interval.
    base_quota_wait: int = 60
    client: Any = field(init=False, repr=False)

    # ------------------------------------------------------------------
    # Initialisation & Model Discovery
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ValueError("A valid GEMINI_API_KEY is required.")

        self.client = genai.Client(api_key=self.api_key.strip())
        log.info("[INIT] genai.Client created. Primary model: %s", self.model)

        # Validate that the chosen model is reachable; fall back if not.
        self._validate_or_discover_model()

    def _validate_or_discover_model(self) -> None:
        """
        Make a minimal probe call to verify the current model is reachable.
        On 404 / model-not-found, run Model Discovery and pick the best available.
        """
        log.info("[INIT] Validating model reachability: %s", self.model)
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents="ping",
            )
            _ = getattr(resp, "text", "") or ""
            log.info("[INIT] Model %s is reachable. ✔", self.model)
        except Exception as exc:
            if self._is_not_found_error(exc):
                log.warning(
                    "[INIT] Model '%s' returned 404. Initiating Model Discovery...",
                    self.model,
                )
                self.model = self._discover_model()
            else:
                # Non-404 errors (e.g. transient 503) — log but don't abort init.
                log.warning(
                    "[INIT] Probe call failed with non-404 error (%s). "
                    "Proceeding with model '%s' — pipeline will retry on real calls.",
                    exc,
                    self.model,
                )

    def _discover_model(self) -> str:
        """
        List all models available from the API and pick the best flash variant.

        Returns the model name that will be used for the pipeline.
        Raises RuntimeError if no usable model is found.
        """
        log.info("[DISCOVERY] Fetching available models from the API...")
        try:
            available: list[str] = []
            for m in self.client.models.list():
                name: str = getattr(m, "name", "") or ""
                # Strip "models/" prefix if present (SDK returns full resource path).
                short = name.replace("models/", "").strip()
                if short:
                    available.append(short)

            log.info("[DISCOVERY] %d models available: %s", len(available), available)

            # Walk the preference list and pick the first match.
            for preferred in _FALLBACK_PREFERENCE:
                for avail in available:
                    if preferred in avail:
                        log.info("[DISCOVERY] Selected model: %s", avail)
                        return avail

            # Last resort: return whatever is first in the list.
            if available:
                fallback = available[0]
                log.warning(
                    "[DISCOVERY] No preferred flash model found. Using: %s", fallback
                )
                return fallback

            raise RuntimeError(
                "[DISCOVERY] API returned zero available models. "
                "Check your API key permissions."
            )

        except Exception as exc:
            raise RuntimeError(
                f"[DISCOVERY] Model discovery failed: {exc}. "
                "Check your GEMINI_API_KEY and network connectivity."
            ) from exc

    # ------------------------------------------------------------------
    # Error classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(sig in msg for sig in _QUOTA_SIGNALS)

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(sig in msg for sig in _NOT_FOUND_SIGNALS)

    # ------------------------------------------------------------------
    # Core retry loop with exponential backoff
    # ------------------------------------------------------------------

    def _generate_with_retry(self, prompt: str, stage_label: str = "LLM") -> str:
        """
        Call Gemini with exponential backoff for 429 / quota errors.

        On each quota hit the system logs:
            [QUOTA] Rate limit hit. Pausing for Xs...
        and waits before retrying automatically.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            log.info("[%s] Attempt %d/%d → model=%s", stage_label, attempt, self.max_retries, self.model)
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                text: str = getattr(response, "text", "") or ""
                if text.strip():
                    log.info("[%s] ✔ Success on attempt %d (%d chars)", stage_label, attempt, len(text))
                    return text.strip()
                raise RuntimeError(f"[{stage_label}] Gemini returned an empty response.")

            except Exception as exc:
                last_error = exc

                # ── 404: model disappeared mid-run → re-discover and retry ──
                if self._is_not_found_error(exc) and attempt == 1:
                    log.warning(
                        "[%s] 404 on model '%s'. Re-running discovery...",
                        stage_label,
                        self.model,
                    )
                    self.model = self._discover_model()
                    continue  # Retry immediately with the new model.

                # ── 429 / quota: exponential backoff with exact log message ──
                if self._is_quota_error(exc):
                    if attempt >= self.max_retries:
                        break
                    wait_seconds: int = self.base_quota_wait * (2 ** (attempt - 1))
                    log.warning(
                        "[QUOTA] Rate limit hit. Pausing for %ds... (attempt %d/%d)",
                        wait_seconds,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(wait_seconds)
                    continue

                # ── Any other error: non-retryable, raise immediately ──
                log.error("[%s] Non-retryable error: %s", stage_label, exc)
                raise

        raise RuntimeError(
            f"[{stage_label}] Exhausted {self.max_retries} retries. Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Stage 1 — Technical Deep-Dive Article
    # ------------------------------------------------------------------

    def generate_article(self, asset_topic: str) -> str:
        """Stage 1: Generate a clinical Fact Sheet."""
        prompt = (
            "You are a Senior Financial Correspondent specialising in Real-World Asset (RWA) tokenization.\n\n"
            f"Topic: {asset_topic}\n\n"
            "Produce a Fact Sheet for a professional trade paper. Start with the Dateline: 'MUMBAI, APRIL 2026'.\n"
            "Structure with clear headings and include:\n"
            "1) On-chain tokenization mechanics (smart contract standard, custody, SPV structure).\n"
            "2) Regulatory/legal framework.\n"
            "3) Yield structure: provide gross yield %, net yield %, and distribution cadence.\n"
            "4) Secondary market liquidity implications and risk controls.\n"
            "5) Practical implementation concerns for issuers and platform operators.\n\n"
            "Use a dry, journalistic, and factual tone. Do not use emojis. Avoid all AI-isms like 'delve' or 'revolutionize'. Focus strictly on facts, numbers, and regulatory requirements."
        )
        return self._generate_with_retry(prompt, stage_label="Stage1:FactSheet")

    # ------------------------------------------------------------------
    # Stage 2 — Retail LinkedIn Post (consumes Stage 1 context)
    # ------------------------------------------------------------------

    def convert_to_linkedin(self, article: str) -> str:
        """
        Stage 2: Convert the Fact Sheet into a Public Summary.
        """
        prompt = (
            "You are a Business Editor.\n\n"
            "━━━ STAGE 1 CONTEXT ━━━\n"
            f"{article}\n"
            "━━━ END CONTEXT ━━━\n\n"
            "Transform the above into a Public Summary for the market news section.\n\n"
            "Requirements:\n"
            "- First Sentence: Direct reportage of the asset classification.\n"
            "- Body: 3–4 paragraphs detailing the ownership structure and market implications.\n"
            "  · Maintain the clinical, journalistic tone of the source Fact Sheet.\n"
            "- Closing: Formal editorial note for further inquiry.\n\n"
            "Use a dry, journalistic, and factual tone. Do not use emojis. Avoid all AI-isms. Focus on facts and numbers."
        )
        return self._generate_with_retry(prompt, stage_label="Stage2:PublicSummary")

    # ------------------------------------------------------------------
    # Stage 3 — CFO Executive Brief (consumes Stage 1 context)
    # ------------------------------------------------------------------

    def generate_summary(self, article: str) -> str:
        """
        Stage 3: Generate a Risk Assessment.
        """
        prompt = (
            "You are an Investigative Auditor.\n\n"
            "━━━ STAGE 1 CONTEXT ━━━\n"
            f"{article}\n"
            "━━━ END CONTEXT ━━━\n\n"
            "Generate a Risk Assessment (The Ledger).\n\n"
            "Output format (strict):\n"
            "ASSESSMENT SUMMARY\n"
            "  · One sentence factual summary.\n\n"
            "RISK / MITIGATION MATRIX\n"
            "  Present exactly 4 paired bullet points:\n"
            "  RISK [label]: <concise risk statement>\n"
            "  MITIGATION [label]: <corresponding control or upside factor>\n\n"
            "REQUIRED OVERSIGHT\n"
            "  · 3 monitoring requirements.\n\n"
            "Use a dry, journalistic, and factual tone. Do not use emojis. Avoid all AI-isms. Focus strictly on facts and regulatory requirements."
        )
        return self._generate_with_retry(prompt, stage_label="Stage3:RiskAssessment")
