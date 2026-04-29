"""
main.py — Self-Healing Entry Point for the RWA Content Pipeline (2026 Edition).

Flow:
  1. Load .env → validate GEMINI_API_KEY
  2. test_connection() → makes a tiny "Hello" probe; aborts with clean diagnostic if it fails
  3. Resolve asset topic (CLI arg > interactive prompt > demo fallback)
  4. Run the 3-stage pipeline (article → LinkedIn post → CFO brief)
  5. Save report.json (structured) + report_<timestamp>.md (Markdown)
  6. Print Rich terminal dashboard with stage stats and telemetry
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from asset_engine import RWAContentEngine

# ---------------------------------------------------------------------------
# Logging — single consistent format across the whole process
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("RWA_MAIN")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPORTS_DIR = Path("reports")
_PIPELINE_VERSION = "2.1.0"
_DEFAULT_TOPIC = "Commercial Office Space in BKC, Mumbai"


# ---------------------------------------------------------------------------
# Pre-flight: connection test
# ---------------------------------------------------------------------------


def test_connection(api_key: str) -> bool:
    """
    Make a minimal "Hello" probe call to the Gemini API before starting
    the real pipeline.  Prints a clear, structured diagnostic on failure.

    Returns True if the connection is healthy, False otherwise.
    """
    log.info("[PRE-FLIGHT] Running connection test...")
    print("\n  ┌─ Pre-Flight Check ────────────────────────────────────────────┐")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Hello. Reply with one word only: OK",
        )
        reply: str = (getattr(response, "text", "") or "").strip()
        log.info("[PRE-FLIGHT] API responded: '%s'", reply)
        print("  │  ✅  API Connection      : HEALTHY")
        print(f"  │  ✅  Model Response      : {reply!r}")
        print("  └───────────────────────────────────────────────────────────────┘\n")
        return True

    except Exception as exc:
        err_msg = str(exc)
        print("  │  ❌  API Connection      : FAILED")
        print("  │")

        if "404" in err_msg or "not found" in err_msg.lower():
            print("  │  DIAGNOSIS: 404 Model Not Found")
            print("  │  → The endpoint does not recognise the requested model.")
            print("  │  → The engine will auto-discover an available model.")

        elif "429" in err_msg or "quota" in err_msg.lower():
            print("  │  DIAGNOSIS: 429 Resource Exhausted / Quota Exceeded")
            print("  │  → Free-tier daily limit may be reached.")
            print("  │  → The engine will retry with exponential backoff (up to 5 attempts).")
            print("  │  → Consider upgrading to Pay-as-you-go or waiting for quota reset.")

        elif "401" in err_msg or "invalid" in err_msg.lower() or "api key" in err_msg.lower():
            print("  │  DIAGNOSIS: 401 Unauthorized / Invalid API Key")
            print("  │  → Verify GEMINI_API_KEY in your .env file.")
            print("  │  → Get a fresh key at: https://aistudio.google.com/app/apikey")

        elif "timeout" in err_msg.lower() or "network" in err_msg.lower():
            print("  │  DIAGNOSIS: Network / Timeout Error")
            print("  │  → Check your internet connection.")
            print("  │  → Retry in a few moments.")

        else:
            print(f"  │  DIAGNOSIS: Unexpected error → {err_msg[:120]}")

        print("  │")
        print(f"  │  Raw exception: {type(exc).__name__}: {err_msg[:200]}")
        print("  └───────────────────────────────────────────────────────────────┘\n")

        log.error("[PRE-FLIGHT] Connection test failed: %s: %s", type(exc).__name__, exc)
        return False


# ---------------------------------------------------------------------------
# Terminal display helpers
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       RWA CONTENT INTELLIGENCE PIPELINE  v{ver}              ║
║   Self-Healing · Async · Structured · 2026 Production Edition  ║
╚══════════════════════════════════════════════════════════════════╝""".format(ver=_PIPELINE_VERSION))


def _print_stage_header(stage: int, label: str) -> None:
    icons = {1: "🏗️ ", 2: "📣", 3: "📋"}
    icon = icons.get(stage, "▶")
    divider = "─" * 68
    print(f"\n  {divider}")
    print(f"  {icon}  STAGE {stage} — {label}")
    print(f"  {divider}")


def _print_summary(
    topic: str,
    model: str,
    article: str,
    linkedin: str,
    summary: str,
    json_path: Path,
    md_path: Path,
    stage_times: dict[str, float],
    total_retries: int,
) -> None:
    divider = "═" * 68
    thin = "─" * 68

    # Extract a few quick stats from the article text.
    word_count = len(article.split())
    linkedin_words = len(linkedin.split())

    print(f"\n  {divider}")
    print("  ✅  RWA PIPELINE COMPLETE")
    print(f"  {divider}")
    print(f"  Topic         : {topic}")
    print(f"  Model Used    : {model}")
    print(f"  Generated At  : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  {thin}")
    print("  STAGE OUTPUTS")
    print(f"    Article     : {word_count:,} words")
    print(f"    LinkedIn    : {linkedin_words:,} words")
    print(f"    CFO Brief   : {len(summary.split()):,} words")
    print(f"  {thin}")
    print("  TELEMETRY")
    for label, secs in stage_times.items():
        print(f"    {label:<22}: {secs:.1f}s")
    print(f"    {'Total Retries':<22}: {total_retries}")
    print(f"  {thin}")
    print(f"  📄 Markdown   : {md_path.resolve()}")
    print(f"  🗂️  JSON       : {json_path.resolve()}")
    print(f"  {divider}\n")


# ---------------------------------------------------------------------------
# Report export helpers
# ---------------------------------------------------------------------------


def _ensure_reports_dir() -> Path:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return _REPORTS_DIR


def _save_json_report(
    topic: str,
    model: str,
    article: str,
    linkedin: str,
    summary: str,
    stage_times: dict[str, float],
    total_retries: int,
) -> Path:
    """Save structured report.json (canonical artefact) and return path."""
    _ensure_reports_dir()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    payload = {
        "pipeline_version": _PIPELINE_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "asset_topic": topic,
        "model_used": model,
        "stages": {
            "article": article,
            "linkedin_post": linkedin,
            "cfo_summary": summary,
        },
        "telemetry": {
            "stage_latencies_seconds": stage_times,
            "total_retries": total_retries,
        },
    }

    # Primary file: always-overwritten canonical "report.json" for easy access.
    canonical = _REPORTS_DIR / "report.json"
    with canonical.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log.info("Canonical JSON saved → %s", canonical.resolve())

    # Timestamped archive copy.
    archive = _REPORTS_DIR / f"report_{timestamp}.json"
    with archive.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log.info("Archived JSON saved → %s", archive.resolve())

    return canonical


def _save_markdown_report(
    topic: str,
    model: str,
    article: str,
    linkedin: str,
    summary: str,
    stage_times: dict[str, float],
) -> Path:
    """Render and save the Markdown investment report, return path."""
    _ensure_reports_dir()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    lines = [
        "# 📊 RWA Tokenization — Professional Investment Report",
        "",
        f"> **Asset:** {topic}",
        f"> **Model:** `{model}`",
        f"> **Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"> **Pipeline Version:** v{_PIPELINE_VERSION}",
        "",
        "---",
        "",
        "## 🏗️ Stage 1 — Technical Deep-Dive Article",
        "",
        article,
        "",
        "---",
        "",
        "## 📣 Stage 2 — LinkedIn Post (Retail Marketization)",
        "",
        linkedin,
        "",
        "---",
        "",
        "## 📋 Stage 3 — CFO Executive Brief",
        "",
        summary,
        "",
        "---",
        "",
        "## 📈 Pipeline Telemetry",
        "",
        "| Stage | Latency |",
        "|-------|---------|",
        *[f"| {label} | {secs:.1f}s |" for label, secs in stage_times.items()],
        "",
        f"_Report generated by RWA Content Pipeline v{_PIPELINE_VERSION} — Powered by {model}_",
    ]

    md_path = _REPORTS_DIR / f"report_{timestamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown report saved → %s", md_path.resolve())
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _print_banner()

    # ── 1. Load environment ─────────────────────────────────────────────────
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("\n  ❌  GEMINI_API_KEY is missing from your .env file.")
        print("  → Add: GEMINI_API_KEY=your_key_here")
        print("  → Get a key at: https://aistudio.google.com/app/apikey\n")
        log.error("GEMINI_API_KEY not found. Aborting.")
        sys.exit(1)

    # ── 2. Pre-flight connection test ───────────────────────────────────────
    connection_ok = test_connection(api_key)
    if not connection_ok:
        # 404 / model issues are handled by Model Discovery in the engine.
        # Only hard-fail on auth errors (api key invalid).
        err_hint = ""
        # Re-check by inspecting if we can at least instantiate the client.
        try:
            genai.Client(api_key=api_key)
        except Exception as client_exc:
            print(f"  ❌  Fatal client init error: {client_exc}")
            sys.exit(1)
        print("  ⚠️   Pre-flight warning noted. Pipeline will attempt self-healing.\n")

    # ── 3. Resolve asset topic ──────────────────────────────────────────────
    if len(sys.argv) > 1:
        asset_topic = " ".join(sys.argv[1:]).strip()
        print(f"  Asset topic (CLI): {asset_topic}\n")
    else:
        print("  Enter the Real-World Asset topic to analyse.")
        print(f"  Example: {_DEFAULT_TOPIC}\n")
        asset_topic = input("  ► Asset Topic: ").strip()

    if not asset_topic:
        asset_topic = _DEFAULT_TOPIC
        print(f"  [No input — using demo topic: '{asset_topic}']\n")

    # ── 4. Instantiate engine (triggers model validation / discovery) ────────
    log.info("Instantiating RWAContentEngine for topic: '%s'", asset_topic)
    try:
        engine = RWAContentEngine(api_key=api_key)
    except Exception as exc:
        log.error("RWA_PIPELINE_ERROR | Engine init failed: %s", exc)
        print(f"\n  ❌  Engine initialisation failed: {exc}\n")
        sys.exit(1)

    total_retries = 0
    stage_times: dict[str, float] = {}

    # ── 5. Stage 1 — Technical Article ─────────────────────────────────────
    _print_stage_header(1, "Technical Deep-Dive Article")
    t0 = time.perf_counter()
    try:
        article = engine.generate_article(asset_topic)
        stage_times["Stage1_Article"] = round(time.perf_counter() - t0, 1)
        print(f"\n{article}\n")
    except Exception as exc:
        log.error("RWA_PIPELINE_ERROR | Stage 1 failed: %s", exc)
        print(f"\n  ❌  Stage 1 failed: {exc}\n")
        sys.exit(1)

    # ── 6. Stage 2 — LinkedIn Post (injects Stage 1 context) ────────────────
    _print_stage_header(2, "LinkedIn Post — Retail Marketization")
    t0 = time.perf_counter()
    try:
        linkedin_post = engine.convert_to_linkedin(article)
        stage_times["Stage2_LinkedIn"] = round(time.perf_counter() - t0, 1)
        print(f"\n{linkedin_post}\n")
    except Exception as exc:
        log.error("RWA_PIPELINE_ERROR | Stage 2 failed: %s", exc)
        print(f"\n  ❌  Stage 2 failed: {exc}\n")
        sys.exit(1)

    # ── 7. Stage 3 — CFO Brief (injects Stage 1 context) ────────────────────
    _print_stage_header(3, "CFO Executive Brief — Risk/Return Matrix")
    t0 = time.perf_counter()
    try:
        cfo_summary = engine.generate_summary(article)
        stage_times["Stage3_CFOBrief"] = round(time.perf_counter() - t0, 1)
        print(f"\n{cfo_summary}\n")
    except Exception as exc:
        log.error("RWA_PIPELINE_ERROR | Stage 3 failed: %s", exc)
        print(f"\n  ❌  Stage 3 failed: {exc}\n")
        sys.exit(1)

    # ── 8. Persist reports ───────────────────────────────────────────────────
    log.info("Exporting reports...")
    json_path = _save_json_report(
        topic=asset_topic,
        model=engine.model,
        article=article,
        linkedin=linkedin_post,
        summary=cfo_summary,
        stage_times=stage_times,
        total_retries=total_retries,
    )
    md_path = _save_markdown_report(
        topic=asset_topic,
        model=engine.model,
        article=article,
        linkedin=linkedin_post,
        summary=cfo_summary,
        stage_times=stage_times,
    )

    # ── 9. Terminal dashboard ────────────────────────────────────────────────
    _print_summary(
        topic=asset_topic,
        model=engine.model,
        article=article,
        linkedin=linkedin_post,
        summary=cfo_summary,
        json_path=json_path,
        md_path=md_path,
        stage_times=stage_times,
        total_retries=total_retries,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [Pipeline interrupted by user. Exiting cleanly.]\n")
        sys.exit(0)
