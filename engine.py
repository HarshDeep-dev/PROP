"""
engine.py — RWAArchitect: Core Async Intelligence Engine for RWA Content Pipeline.

Architecture:
  · Async-first (asyncio) — all LLM calls are non-blocking coroutines.
  · Exponential backoff with jitter for 429 / Resource-Exhausted resilience.
  · StructuredOutputParser converts raw LLM text → validated Pydantic models.
  · State management: Stage 1 article is injected into Stage 2 & 3 prompts.
  · Python logging tracks every pipeline heartbeat event.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from typing import Any, TypeVar

from google import genai

from schema import (
    LinkedInCTA,
    PipelineMetadata,
    RiskLevel,
    RiskReturnVector,
    RWAArticle,
    RWAContent,
    RWAExecutiveBrief,
    RWALegalFramework,
    RWALinkedInPost,
    YieldStructure,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

RWA_PIPELINE_LOGGER = "RWA_PIPELINE"
LOG_FORMAT = (
    "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
)
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(RWA_PIPELINE_LOGGER)

# Custom log-level aliases for semantic clarity in pipeline heartbeat tracking.
RWA_PIPELINE_START = logging.INFO
RWA_PIPELINE_LLM_CALL = logging.INFO
RWA_PIPELINE_RETRY = logging.WARNING
RWA_PIPELINE_SUCCESS = logging.INFO
RWA_PIPELINE_ERROR = logging.ERROR

# ---------------------------------------------------------------------------
# Typed aliases
# ---------------------------------------------------------------------------

T = TypeVar("T")

# ---------------------------------------------------------------------------
# StructuredOutputParser
# ---------------------------------------------------------------------------


class StructuredOutputParser:
    """
    Converts raw LLM text output into validated Pydantic model instances.

    Strategy:
      1. Try to locate a JSON block in the response (```json ... ``` fence or raw {}).
      2. Parse with json.loads.
      3. Construct and validate the target Pydantic model.
      4. On failure, raise a descriptive RWAPipelineParseError.
    """

    class RWAPipelineParseError(RuntimeError):
        """Raised when the LLM response cannot be parsed into the target schema."""

    _JSON_FENCE_RE = re.compile(
        r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
        re.DOTALL | re.IGNORECASE,
    )
    _BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)

    @classmethod
    def extract_json(cls, raw_text: str) -> dict[str, Any]:
        """Pull the first JSON object from arbitrary LLM text."""
        # Try fenced block first.
        fence_match = cls._JSON_FENCE_RE.search(raw_text)
        if fence_match:
            return json.loads(fence_match.group(1))

        # Fallback: bare JSON object.
        bare_match = cls._BARE_JSON_RE.search(raw_text)
        if bare_match:
            return json.loads(bare_match.group(1))

        raise cls.RWAPipelineParseError(
            "No JSON object found in LLM response. Raw text (first 500 chars):\n"
            + raw_text[:500]
        )

    @classmethod
    def parse(cls, raw_text: str, model_cls: type[T]) -> T:
        """Parse raw LLM text into a validated Pydantic model of type `model_cls`."""
        try:
            data = cls.extract_json(raw_text)
            return model_cls.model_validate(data)  # type: ignore[attr-defined]
        except (json.JSONDecodeError, ValueError) as exc:
            raise cls.RWAPipelineParseError(
                f"Failed to parse LLM output into {model_cls.__name__}: {exc}\n"
                f"Raw text: {raw_text[:800]}"
            ) from exc


# ---------------------------------------------------------------------------
# RWAArchitect — Core async pipeline class
# ---------------------------------------------------------------------------


class RWAArchitect:
    """
    Orchestrates the 3-stage RWA Content Intelligence Pipeline.

    Stage 1: Technical Deep-Dive Article   → RWAArticle
    Stage 2: Retail LinkedIn Post          → RWALinkedInPost  (uses Stage 1 context)
    Stage 3: CFO Executive Brief           → RWAExecutiveBrief (uses Stage 1 context)
    """

    _MAX_RETRIES: int = 6
    _BASE_BACKOFF_SECONDS: float = 8.0
    _MAX_BACKOFF_SECONDS: float = 300.0
    _JITTER_FACTOR: float = 0.3  # ±30% random jitter on each backoff interval.

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("A valid GEMINI_API_KEY is required to initialise RWAArchitect.")

        self._model = model
        self._client = genai.Client(api_key=api_key.strip())
        self._parser = StructuredOutputParser()
        self._total_retries: int = 0

        log.log(RWA_PIPELINE_START, "RWAArchitect initialised | model=%s", model)

    # ------------------------------------------------------------------
    # Resilience: Exponential Backoff Wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Detect 429 / Resource Exhausted errors from any SDK version."""
        msg = str(exc).lower()
        return any(
            token in msg
            for token in ("429", "resource exhausted", "rate limit", "quota exceeded", "quota")
        )

    def _compute_backoff(self, attempt: int) -> float:
        """Exponential backoff with ±jitter, capped at _MAX_BACKOFF_SECONDS."""
        raw = self._BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
        jitter = raw * self._JITTER_FACTOR * (random.random() * 2 - 1)  # ±jitter
        return min(raw + jitter, self._MAX_BACKOFF_SECONDS)

    async def _call_llm_with_backoff(self, prompt: str, stage_label: str) -> str:
        """
        Async wrapper around the Gemini client with exponential backoff.
        Runs the synchronous SDK call in a thread-pool executor to stay non-blocking.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            log.log(
                RWA_PIPELINE_LLM_CALL,
                "[%s] LLM call | attempt=%d/%d",
                stage_label,
                attempt,
                self._MAX_RETRIES,
            )
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda p=prompt: self._client.models.generate_content(
                        model=self._model,
                        contents=p,
                    ),
                )
                text: str = getattr(response, "text", "") or ""
                if not text.strip():
                    raise RuntimeError(
                        f"[{stage_label}] Gemini returned an empty response on attempt {attempt}."
                    )

                log.log(
                    RWA_PIPELINE_SUCCESS,
                    "[%s] LLM call SUCCESS | chars=%d",
                    stage_label,
                    len(text),
                )
                return text.strip()

            except Exception as exc:
                last_exc = exc

                if not self._is_rate_limit_error(exc):
                    log.log(
                        RWA_PIPELINE_ERROR,
                        "[%s] Non-retryable error on attempt %d: %s",
                        stage_label,
                        attempt,
                        exc,
                    )
                    raise

                if attempt >= self._MAX_RETRIES:
                    break

                wait = self._compute_backoff(attempt)
                self._total_retries += 1
                log.log(
                    RWA_PIPELINE_RETRY,
                    "[%s] 429/Resource Exhausted | retry=%d | backoff=%.1fs | err=%s",
                    stage_label,
                    attempt,
                    wait,
                    exc,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"[{stage_label}] Exhausted {self._MAX_RETRIES} retries due to rate limiting."
        ) from last_exc

    # ------------------------------------------------------------------
    # Prompt Builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_article_prompt(asset_topic: str) -> str:
        return f"""
You are a Senior Financial Correspondent specialising in Real-World Asset (RWA) tokenization.

TASK: Generate a Fact Sheet for a professional trade paper on:
"{asset_topic}"

Return your response as a SINGLE valid JSON object matching this schema EXACTLY:

{{
  "title": "<Professional article title>",
  "executive_summary": "<2-3 sentence executive overview>",
  "tokenization_mechanics": "<Step-by-step tokenization process, min 200 chars>",
  "legal_framework": {{
    "jurisdiction": "<Primary legal jurisdiction, e.g. 'India – RERA / SEBI'>",
    "regulation_type": "<One of: RERA, SEBI, SEC, MAS, FSCA, OTHER>",
    "compliance_notes": "<Key compliance obligations, min 50 chars>",
    "smart_contract_standard": "<e.g. ERC-3643 or ERC-1400>"
  }},
  "yield_structure": {{
    "gross_yield_pct": <float, e.g. 8.5>,
    "net_yield_pct": <float, e.g. 6.8>,
    "distribution_frequency": "<Monthly | Quarterly | Annually>",
    "liquidity_premium_bps": <integer basis points, e.g. 150>
  }},
  "secondary_market_analysis": "<Liquidity, AMM/orderbook, secondary risk, min 100 chars>",
  "risk_factors": [
    "<Risk 1, min 20 chars>",
    "<Risk 2, min 20 chars>",
    "<Risk 3, min 20 chars>",
    "<Risk 4, min 20 chars>"
  ],
  "full_body": "<Complete 500+ word markdown article with clear headings>",
  "word_count": 0
}}

CRITICAL RULES:
- Return ONLY the JSON object. No prose before or after.
- Use realistic, data-driven figures for the specific asset class and geography.
- Start full_body with the Dateline: 'MUMBAI, APRIL 2026'.
- Use a dry, journalistic, and factual tone. Do not use emojis. Avoid all AI-isms. Focus strictly on facts, numbers, and regulatory requirements.
- full_body must be at least 400 words of substantive factual content.
""".strip()

    @staticmethod
    def _build_linkedin_prompt(asset_topic: str, article: RWAArticle) -> str:
        return f"""
You are a Business Editor.

CONTEXT — Stage 1 Fact Sheet (use this for narrative consistency):
Title: {article.title}
Executive Summary: {article.executive_summary}
Gross Yield: {article.yield_structure.gross_yield_pct}% | Net Yield: {article.yield_structure.net_yield_pct}%
Jurisdiction: {article.legal_framework.jurisdiction}
Key Risks: {'; '.join(article.risk_factors[:3])}

TASK: Convert the above into a Public Summary for the market news section on:
"{asset_topic}"

Return your response as a SINGLE valid JSON object matching this schema EXACTLY:

{{
  "hook": "<Single punchy scroll-stopping line, 30-180 chars>",
  "value_body": "<Core fractional ownership value proposition, 3-5 paragraphs, min 200 chars>",
  "cta": {{
    "cta_text": "<Actionable call-to-action sentence, min 25 chars>",
    "hashtags": ["RWATokenization", "FractionalOwnership", "<3-5 more relevant tags>"]
  }},
  "estimated_engagement_score": <integer 1-10>,
  "full_post_text": "<Complete ready-to-post LinkedIn text including hook, body, CTA and hashtags>"
}}

CRITICAL RULES:
- Return ONLY the JSON. No prose before or after.
- full_post_text must be a plain, complete summary (min 250 words).
- Use a dry, journalistic, and factual tone. Do not use emojis. Avoid all AI-isms. Focus on facts and numbers.
""".strip()

    @staticmethod
    def _build_brief_prompt(asset_topic: str, article: RWAArticle) -> str:
        return f"""
You are an Investigative Auditor.

CONTEXT — Stage 1 Fact Sheet (use this for narrative consistency):
Title: {article.title}
Executive Summary: {article.executive_summary}
Gross/Net Yield: {article.yield_structure.gross_yield_pct}% / {article.yield_structure.net_yield_pct}%
Smart Contract Standard: {article.legal_framework.smart_contract_standard}
Risk Factors: {'; '.join(article.risk_factors)}
Secondary Market: {article.secondary_market_analysis}

TASK: Produce a Risk Assessment (The Ledger) for:
"{asset_topic}"

Return your response as a SINGLE valid JSON object matching this schema EXACTLY:

{{
  "asset_classification": "<Asset class and sub-class, e.g. 'Commercial Real Estate – Grade-A Office'>",
  "investment_thesis": "<2-sentence boardroom-ready thesis, min 100 chars>",
  "risk_return_matrix": [
    {{
      "dimension": "<Strategic dimension, e.g. 'Market Liquidity'>",
      "risk_statement": "<Concise risk for this dimension, min 40 chars>",
      "return_statement": "<Corresponding upside/return opportunity, min 40 chars>",
      "risk_level": "<One of: LOW, MODERATE, HIGH, CRITICAL>",
      "mitigation_lever": "<Tactical or structural mitigation, min 30 chars>"
    }},
    {{ ... }},
    {{ ... }}
  ],
  "overall_risk_rating": "<One of: LOW, MODERATE, HIGH, CRITICAL>",
  "recommended_allocation_pct": <float, e.g. 5.0>,
  "key_watch_items": [
    "<Watch item 1>",
    "<Watch item 2>",
    "<Watch item 3>"
  ],
  "full_brief_text": "<Complete 400+ word board-ready prose executive brief>"
}}

CRITICAL RULES:
- Return ONLY the JSON. No prose before or after.
- Include at least 4 risk_return_matrix entries.
- Figures must align with the Stage 1 Fact Sheet data.
- full_brief_text must be at least 300 words of substantive assessment prose.
- Use a dry, journalistic, and factual tone. Do not use emojis. Avoid all AI-isms. Focus strictly on facts and regulatory requirements.
""".strip()

    # ------------------------------------------------------------------
    # Stage Runners
    # ------------------------------------------------------------------

    async def _run_stage_1(self, asset_topic: str) -> tuple[RWAArticle, float]:
        log.log(RWA_PIPELINE_START, "▶ Stage 1 START | topic='%s'", asset_topic)
        t0 = time.perf_counter()
        raw = await self._call_llm_with_backoff(
            self._build_article_prompt(asset_topic),
            stage_label="Stage1:Article",
        )
        article = StructuredOutputParser.parse(raw, RWAArticle)
        # Populate word count post-parse.
        article.word_count = len(article.full_body.split())
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.log(
            RWA_PIPELINE_SUCCESS,
            "✔ Stage 1 DONE | title='%s' | words=%d | latency=%.0fms",
            article.title,
            article.word_count,
            elapsed_ms,
        )
        return article, elapsed_ms

    async def _run_stage_2(
        self, asset_topic: str, article: RWAArticle
    ) -> tuple[RWALinkedInPost, float]:
        log.log(RWA_PIPELINE_START, "▶ Stage 2 START | injecting Stage 1 context")
        t0 = time.perf_counter()
        raw = await self._call_llm_with_backoff(
            self._build_linkedin_prompt(asset_topic, article),
            stage_label="Stage2:LinkedIn",
        )
        post = StructuredOutputParser.parse(raw, RWALinkedInPost)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.log(
            RWA_PIPELINE_SUCCESS,
            "✔ Stage 2 DONE | engagement_score=%d | latency=%.0fms",
            post.estimated_engagement_score,
            elapsed_ms,
        )
        return post, elapsed_ms

    async def _run_stage_3(
        self, asset_topic: str, article: RWAArticle
    ) -> tuple[RWAExecutiveBrief, float]:
        log.log(RWA_PIPELINE_START, "▶ Stage 3 START | injecting Stage 1 context")
        t0 = time.perf_counter()
        raw = await self._call_llm_with_backoff(
            self._build_brief_prompt(asset_topic, article),
            stage_label="Stage3:CFOBrief",
        )
        brief = StructuredOutputParser.parse(raw, RWAExecutiveBrief)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.log(
            RWA_PIPELINE_SUCCESS,
            "✔ Stage 3 DONE | risk_rating=%s | latency=%.0fms",
            brief.overall_risk_rating.value,
            elapsed_ms,
        )
        return brief, elapsed_ms

    # ------------------------------------------------------------------
    # Public Entry Point
    # ------------------------------------------------------------------

    async def execute_pipeline(self, asset_topic: str) -> RWAContent:
        """
        Execute the full 3-stage RWA Content Pipeline for the given asset topic.

        Stages run sequentially so that Stage 1 output can be injected as
        grounding context into Stages 2 and 3 (state management).

        Returns a fully validated RWAContent aggregate model.
        """
        run_id = str(uuid.uuid4())
        log.log(
            RWA_PIPELINE_START,
            "═══ RWA PIPELINE START | run_id=%s | topic='%s' ═══",
            run_id,
            asset_topic,
        )
        pipeline_t0 = time.perf_counter()
        self._total_retries = 0

        # Stage 1 — must complete before 2 & 3.
        article, s1_ms = await self._run_stage_1(asset_topic)

        # Stages 2 & 3 share Stage 1 context but are independent of each other.
        # Run them concurrently for throughput.
        (post, s2_ms), (brief, s3_ms) = await asyncio.gather(
            self._run_stage_2(asset_topic, article),
            self._run_stage_3(asset_topic, article),
        )

        total_ms = (time.perf_counter() - pipeline_t0) * 1000

        metadata = PipelineMetadata(
            asset_topic=asset_topic,
            run_id=run_id,
            model_id=self._model,
            stage_latencies_ms={
                "Stage1_Article": round(s1_ms, 1),
                "Stage2_LinkedIn": round(s2_ms, 1),
                "Stage3_CFOBrief": round(s3_ms, 1),
                "Total": round(total_ms, 1),
            },
            total_retry_count=self._total_retries,
        )

        log.log(
            RWA_PIPELINE_SUCCESS,
            "═══ RWA PIPELINE COMPLETE | run_id=%s | total=%.0fms | retries=%d ═══",
            run_id,
            total_ms,
            self._total_retries,
        )

        return RWAContent(
            metadata=metadata,
            article=article,
            post=post,
            brief=brief,
        )
