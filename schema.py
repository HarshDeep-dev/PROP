"""
schema.py — Pydantic Data Models for the RWA Content Pipeline.

Defines strict, type-safe output schemas for every stage of the
Real-World Asset tokenization pipeline:
  · Stage 1 → RWAArticle        (Fact Sheet)
  · Stage 2 → RWALinkedInPost   (Public Summary)
  · Stage 3 → RWAExecutiveBrief (Risk Assessment)
  · Root    → RWAContent        (Aggregated pipeline output)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared enums & primitive types
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    """Standardised risk classification for executive-grade output."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RegulationType(str, Enum):
    """Regulatory frameworks referenced within the technical article."""

    RERA = "RERA"          # India – Real Estate (Regulation and Development) Act
    SEBI = "SEBI"          # India – Securities and Exchange Board of India
    SEC = "SEC"            # USA  – Securities and Exchange Commission
    MAS = "MAS"            # Singapore – Monetary Authority of Singapore
    FSCA = "FSCA"          # South Africa – Financial Sector Conduct Authority
    OTHER = "OTHER"


# ---------------------------------------------------------------------------
# Stage 1 — Fact Sheet
# ---------------------------------------------------------------------------


class RWALegalFramework(BaseModel):
    """Granular legal and regulatory context parsed from Stage 1 output."""

    jurisdiction: str = Field(
        ...,
        description="Primary legal jurisdiction (e.g., 'India – RERA', 'USA – SEC').",
        min_length=3,
    )
    regulation_type: RegulationType = Field(
        ...,
        description="Applicable regulatory framework enum.",
    )
    compliance_notes: str = Field(
        ...,
        description="Key compliance obligations and enforcement posture.",
        min_length=20,
    )
    smart_contract_standard: str = Field(
        default="ERC-3643",
        description="Token standard used for on-chain compliance (e.g. ERC-3643, ERC-1400).",
    )


class YieldStructure(BaseModel):
    """Institutional-grade yield breakdown for the tokenized asset."""

    gross_yield_pct: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        ...,
        description="Gross annual yield percentage before fees and taxes.",
    )
    net_yield_pct: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        ...,
        description="Net annual yield percentage after fees and taxes.",
    )
    distribution_frequency: str = Field(
        default="Quarterly",
        description="Rental/income distribution cadence (e.g. Monthly, Quarterly).",
    )
    liquidity_premium_bps: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description="Illiquidity premium in basis points above risk-free rate.",
    )


class RWAArticle(BaseModel):
    """
    Stage 1 output: Structured Fact Sheet.
    Used as source context for Stage 2 and Stage 3.
    """

    title: str = Field(
        ...,
        description="Professional article title.",
        min_length=10,
    )
    executive_summary: str = Field(
        ...,
        description="2–3 sentence executive overview of the asset and tokenization thesis.",
        min_length=50,
    )
    tokenization_mechanics: str = Field(
        ...,
        description="Step-by-step explanation of how the RWA is tokenized on-chain.",
        min_length=100,
    )
    legal_framework: RWALegalFramework = Field(
        ...,
        description="Structured legal and regulatory context.",
    )
    yield_structure: YieldStructure = Field(
        ...,
        description="Institutional yield breakdown.",
    )
    secondary_market_analysis: str = Field(
        ...,
        description="Liquidity depth, AMM/orderbook design, and secondary market risk.",
        min_length=80,
    )
    risk_factors: list[str] = Field(
        ...,
        description="Enumerated risk factors (minimum 3, each ≥ 15 chars).",
        min_length=3,
    )
    full_body: str = Field(
        ...,
        description="Complete markdown-formatted article body (≥ 400 words).",
        min_length=400,
    )
    word_count: int = Field(
        default=0,
        description="Auto-computed word count of full_body.",
    )

    @field_validator("word_count", mode="before")
    @classmethod
    def _auto_word_count(cls, v: int) -> int:  # noqa: N805
        # Word count is populated post-init by the engine; accept 0 as sentinel.
        return v

    @field_validator("risk_factors")
    @classmethod
    def _validate_risk_factors(cls, v: list[str]) -> list[str]:
        for i, item in enumerate(v):
            if len(item) < 15:
                raise ValueError(
                    f"risk_factors[{i}] is too short ({len(item)} chars); minimum 15."
                )
        return v


# ---------------------------------------------------------------------------
# Stage 2 — Public Summary
# ---------------------------------------------------------------------------


class LinkedInCTA(BaseModel):
    """Structured Call-To-Action appended to the LinkedIn post."""

    cta_text: str = Field(
        ...,
        description="The actionable call-to-action sentence.",
        min_length=20,
    )
    hashtags: list[str] = Field(
        ...,
        description="3–7 relevant hashtags (without the # prefix).",
        min_length=3,
        max_length=7,
    )

    @field_validator("hashtags")
    @classmethod
    def _strip_hash(cls, v: list[str]) -> list[str]:
        return [tag.lstrip("#").strip() for tag in v]


class RWALinkedInPost(BaseModel):
    """
    Stage 2 output: Public Summary for general audiences.
    Generated with Stage 1 Fact Sheet injected into context.
    """

    hook: str = Field(
        ...,
        description="Single punchy hook line to arrest the scroll.",
        min_length=20,
        max_length=200,
    )
    value_body: str = Field(
        ...,
        description="Core value proposition paragraphs (fractional ownership thesis).",
        min_length=150,
    )
    cta: LinkedInCTA = Field(
        ...,
        description="Structured CTA with hashtag block.",
    )
    estimated_engagement_score: Annotated[int, Field(ge=1, le=10)] = Field(
        default=7,
        description="LLM self-assessed virality score on a 1–10 scale.",
    )
    full_post_text: str = Field(
        ...,
        description="Complete, ready-to-copy LinkedIn post body.",
        min_length=200,
    )


# ---------------------------------------------------------------------------
# Stage 3 — Risk Assessment
# ---------------------------------------------------------------------------


class RiskReturnVector(BaseModel):
    """A single Risk ↔ Return paired data point in the executive matrix."""

    dimension: str = Field(
        ...,
        description="Strategic dimension label (e.g. 'Market Liquidity', 'Regulatory').",
        min_length=5,
    )
    risk_statement: str = Field(
        ...,
        description="Concise risk articulation for this dimension.",
        min_length=30,
    )
    return_statement: str = Field(
        ...,
        description="Corresponding upside/return opportunity.",
        min_length=30,
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Severity classification of the identified risk.",
    )
    mitigation_lever: str = Field(
        ...,
        description="Recommended tactical or structural mitigation.",
        min_length=20,
    )


class RWAExecutiveBrief(BaseModel):
    """
    Stage 3 output: Risk Assessment.
    Risk and mitigation analysis generated with Stage 1 Fact Sheet as context.
    """

    asset_classification: str = Field(
        ...,
        description="Asset class and sub-class (e.g. 'Commercial Real Estate – Grade-A Office').",
        min_length=10,
    )
    investment_thesis: str = Field(
        ...,
        description="2-sentence board-room-ready investment thesis.",
        min_length=80,
    )
    risk_return_matrix: list[RiskReturnVector] = Field(
        ...,
        description="Structured risk/return vectors (minimum 3).",
        min_length=3,
    )
    overall_risk_rating: RiskLevel = Field(
        ...,
        description="Composite portfolio-level risk rating.",
    )
    recommended_allocation_pct: Annotated[float, Field(ge=0.0, le=100.0)] = Field(
        ...,
        description="Suggested portfolio allocation % for institutional mandates.",
    )
    key_watch_items: list[str] = Field(
        ...,
        description="Top 3 items the CFO must monitor post-investment.",
        min_length=3,
        max_length=5,
    )
    full_brief_text: str = Field(
        ...,
        description="Prose executive brief ready for board distribution.",
        min_length=300,
    )


# ---------------------------------------------------------------------------
# Root Model — Aggregated Pipeline Output
# ---------------------------------------------------------------------------


class PipelineMetadata(BaseModel):
    """Provenance and telemetry metadata for the full pipeline run."""

    pipeline_version: str = Field(default="2.0.0")
    sdk_version: str = Field(default="google-genai-2026")
    model_id: str = Field(default="gemini-2.5-flash-lite")
    asset_topic: str = Field(..., description="Raw user-provided asset topic string.")
    run_id: str = Field(..., description="UUID4 run identifier for idempotency.")
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of pipeline completion.",
    )
    stage_latencies_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage wall-clock latency in milliseconds.",
    )
    total_retry_count: int = Field(
        default=0,
        description="Aggregate retry attempts across all stages.",
    )


class RWAContent(BaseModel):
    """
    Root aggregate model — complete output of the RWA Content Pipeline.

    Carries:
      · metadata   → provenance + telemetry
      · article    → Stage 1 structured technical article
      · post       → Stage 2 LinkedIn retail post
      · brief      → Stage 3 CFO executive brief
    """

    metadata: PipelineMetadata
    article: RWAArticle
    post: RWALinkedInPost
    brief: RWAExecutiveBrief

    def to_markdown_report(self) -> str:
        """Render a full-fidelity Markdown investment report from the pipeline output."""
        lines: list[str] = [
            "# 📊 RWA Tokenization — Professional Investment Report",
            "",
            f"> **Asset:** {self.metadata.asset_topic}",
            f"> **Run ID:** `{self.metadata.run_id}`",
            f"> **Generated:** {self.metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"> **Model:** `{self.metadata.model_id}`",
            "",
            "---",
            "",
            "## 🏗️ Stage 1 — Technical Deep-Dive",
            "",
            f"### {self.article.title}",
            "",
            f"**Executive Summary:** {self.article.executive_summary}",
            "",
            self.article.full_body,
            "",
            "#### ⚙️ Yield Structure",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Gross Yield | {self.article.yield_structure.gross_yield_pct:.2f}% |",
            f"| Net Yield | {self.article.yield_structure.net_yield_pct:.2f}% |",
            f"| Distribution | {self.article.yield_structure.distribution_frequency} |",
            f"| Illiquidity Premium | {self.article.yield_structure.liquidity_premium_bps} bps |",
            "",
            "#### ⚠️ Risk Factors",
            *[f"- {r}" for r in self.article.risk_factors],
            "",
            "---",
            "",
            "## 📣 Stage 2 — LinkedIn Post (Retail Marketization)",
            "",
            f"**Hook:** _{self.post.hook}_",
            "",
            self.post.full_post_text,
            "",
            f"**CTA:** {self.post.cta.cta_text}",
            f"**Tags:** {' '.join('#' + t for t in self.post.cta.hashtags)}",
            f"**Virality Score:** {self.post.estimated_engagement_score}/10",
            "",
            "---",
            "",
            "## 📋 Stage 3 — CFO Executive Brief",
            "",
            f"**Asset Class:** {self.brief.asset_classification}",
            f"**Investment Thesis:** {self.brief.investment_thesis}",
            f"**Overall Risk Rating:** `{self.brief.overall_risk_rating.value}`",
            f"**Recommended Allocation:** {self.brief.recommended_allocation_pct:.1f}%",
            "",
            "### Risk/Return Matrix",
            "",
            "| Dimension | Risk | Return | Severity | Mitigation |",
            "|-----------|------|--------|----------|------------|",
            *[
                f"| {v.dimension} | {v.risk_statement} | {v.return_statement} "
                f"| `{v.risk_level.value}` | {v.mitigation_lever} |"
                for v in self.brief.risk_return_matrix
            ],
            "",
            "### Key Watch Items",
            *[f"{i + 1}. {item}" for i, item in enumerate(self.brief.key_watch_items)],
            "",
            self.brief.full_brief_text,
            "",
            "---",
            "",
            "### 📈 Pipeline Telemetry",
            "",
            f"| Stage | Latency |",
            f"|-------|---------|",
            *[
                f"| {stage} | {ms:.0f} ms |"
                for stage, ms in self.metadata.stage_latencies_ms.items()
            ],
            f"| **Total Retries** | {self.metadata.total_retry_count} |",
            "",
            "_Report generated by RWA Content Pipeline v{} — Powered by {}_".format(
                self.metadata.pipeline_version, self.metadata.model_id
            ),
        ]
        return "\n".join(lines)
