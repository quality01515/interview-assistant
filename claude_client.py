"""
Claude client. Streams interview answers using the candidate's resume as cached context.

The resume + role description live in the system prompt with cache_control set,
so we pay the ~1.25x write cost only on the first request of the session and a
small fraction of the input price on every subsequent answer.

Model notes (Claude Opus 5.5, the default):
  - Thinking is always on; `output_config.effort` is the only control for how
    much it thinks, and therefore for time-to-first-token. Live answers use
    cfg.effort ("low"); the Deeper button uses cfg.deep_effort.
  - Thinking tokens count toward `max_tokens` even though their text isn't
    returned, so max_tokens must leave room for thinking plus the answer. The
    answer's LENGTH is set by the system prompt, not by this cap.
  - Safety classifiers can decline a request (HTTP 200, stop_reason
    "refusal"). Every call opts into server-side fallbacks, so a false-positive
    decline is retried on the model Anthropic recommends instead of leaving the
    answer box empty.
"""
from __future__ import annotations

import json
from typing import Iterator, Sequence

import anthropic

from config import Config
from pipeline.answer_prompt import SYSTEM_RULES
from pipeline.context_summary import build_update_prompt
from pipeline.evaluation import (
    EVALUATION_SCHEMA, EVALUATION_USER_PROMPT, InterviewEvaluation,
    format_transcript,
)
from pipeline.types import Turn


# Server-side refusal fallback: on a safety-classifier decline the API re-runs
# the same request on a recommended model inside the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_KW = {"betas": [_FALLBACK_BETA], "fallbacks": "default"}


def _first_text(content) -> str:
    """Text of the first `text` block. Read by type — responses can start with
    (empty) thinking blocks."""
    return next((b.text for b in content if getattr(b, "type", None) == "text"), "")


class ClaudeClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not cfg.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it as an environment variable "
                "or save it via the settings dialog."
            )
        self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self._resume_text: str = ""

    # ── one-time per interview ───────────────────────────────────────────────
    def set_context(
        self,
        resume_text: str,
        job_title: str = "",
        job_description: str = "",
        personal_context: str = "",
    ) -> None:
        self._resume_text = resume_text.strip()
        self.cfg.job_title = job_title
        self.cfg.job_description = job_description
        self.cfg.personal_context = personal_context

    # ── one-time warmup (populate prompt cache + establish HTTPS) ──────────
    def warmup(self) -> None:
        """
        Pre-warm with a `max_tokens=0` request so:
          - The HTTPS connection to api.anthropic.com is established
          - The cached system block (resume + role + personal_context) is
            written to Anthropic's 5-minute prompt cache
        The API runs prefill only and returns no output tokens. Effort must
        match the live answers' effort, or the warmed cache entry won't be the
        one real traffic reads. Failures are swallowed — the first real answer
        will just pay the cache-write cost itself instead.
        """
        if not self._resume_text:
            return
        try:
            self._client.messages.create(
                model=self.cfg.model,
                max_tokens=0,
                system=self._build_system_blocks(),
                messages=[{"role": "user", "content": "warmup"}],
                output_config={"effort": self.cfg.effort},
            )
        except Exception:
            pass

    # ── post-interview evaluation (called when user clicks "End interview") ──
    def evaluate_interview(self, turns: Sequence[Turn]) -> InterviewEvaluation:
        """
        Send the full transcript to Claude and ask for a structured hireability
        verdict. Uses the deep model at high effort for a more careful
        assessment. Returns an InterviewEvaluation with .error set if the call
        fails.
        """
        if not self._resume_text:
            return InterviewEvaluation(
                score=0, verdict="fail", summary="",
                error="No resume context — cannot evaluate.",
            )
        transcript = format_transcript(turns)
        if not transcript.strip():
            return InterviewEvaluation(
                score=0, verdict="fail", summary="",
                error="No transcript to evaluate.",
            )
        try:
            # Non-streaming, so stay at ~16K: room for high-effort thinking plus
            # the JSON, while keeping the request under the SDK's HTTP timeout.
            response = self._client.beta.messages.create(
                model=self.cfg.deep_model,
                max_tokens=16000,
                system=self._build_system_blocks(),
                messages=[{
                    "role": "user",
                    "content": EVALUATION_USER_PROMPT.format(transcript=transcript),
                }],
                output_config={
                    "format": {"type": "json_schema", "schema": EVALUATION_SCHEMA},
                    "effort": "high",
                },
                **_FALLBACK_KW,
            )
            if response.stop_reason == "refusal":
                raise RuntimeError("the model declined to evaluate this transcript")
            data = json.loads(_first_text(response.content))
            return InterviewEvaluation(
                score=int(data.get("score", 0)),
                verdict=str(data.get("verdict", "fail")),
                summary=str(data.get("summary", "")),
                strengths=list(data.get("strengths", []) or []),
                concerns=list(data.get("concerns", []) or []),
                specific_moments=list(data.get("specific_moments", []) or []),
            )
        except Exception as e:
            return InterviewEvaluation(
                score=0, verdict="fail", summary="",
                error=f"Evaluation failed: {e}",
            )

    # ── running context summary (see pipeline/context_summary.py) ─────────────
    def summarize(self, prior_summary: str, new_turns: Sequence[Turn]) -> str:
        """
        Fold `new_turns` (aged out of the rolling transcript window) into a
        compact running summary, so later answers stay consistent with the
        whole interview without resending it verbatim every time. Low effort,
        called off the live-answer path, never blocks a `stream_answer()` call.
        max_tokens leaves room for thinking; the prompt caps the summary itself
        at SUMMARY_MAX_WORDS.
        """
        response = self._client.beta.messages.create(
            model=self.cfg.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": build_update_prompt(prior_summary, new_turns)}],
            output_config={"effort": "low"},
            **_FALLBACK_KW,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined to summarize")
        return _first_text(response.content).strip()

    # ── per-question ─────────────────────────────────────────────────────────
    def stream_answer(
        self,
        turns: Sequence[Turn],
        *,
        deep: bool = False,
        style_hint: str = "",
        summary: str = "",
    ) -> Iterator[str]:
        """
        Yield text chunks of the answer as Claude streams it.

        `turns` is the rolling speaker-labelled transcript. The latest interviewer
        turn is the one being answered.

        `deep=True` swaps to the deep-mode model and effort (cfg.deep_model /
        cfg.deep_effort) for a richer answer.
        `style_hint` is appended to nudge length/tone, e.g. "shorter" or "more technical".
        `summary` is the running summary of everything older than `turns` — see
        pipeline/context_summary.py. Empty string if there's nothing to summarize yet.
        """
        if not self._resume_text:
            raise RuntimeError("Call set_context() with the resume before requesting an answer.")

        system = self._build_system_blocks()
        user_message = self._build_user_message(turns, style_hint=style_hint, summary=summary)
        model = self.cfg.deep_model if deep else self.cfg.model
        effort = self.cfg.deep_effort if deep else self.cfg.effort

        # text_stream yields only answer text; thinking blocks (empty under the
        # default display) are skipped. A caller that stops iterating (a newer
        # answer superseded this one) closes the generator, which exits the
        # `with` and closes the HTTP stream.
        with self._client.beta.messages.stream(
            model=model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            output_config={"effort": effort},
            **_FALLBACK_KW,
        ) as stream:
            for text in stream.text_stream:
                yield text
            if stream.get_final_message().stop_reason == "refusal":
                # Only reached when the fallback model declined too.
                raise RuntimeError("the model declined to answer this question")

    # ── internals ────────────────────────────────────────────────────────────
    def _build_system_blocks(self) -> list[dict]:
        # Render order is tools → system → messages. Stable content first, then
        # the cached resume+role block, then nothing volatile in system.
        role_block = (
            f"<target_role>\n{self.cfg.job_title or '(not specified)'}\n\n"
            f"Job description:\n{self.cfg.job_description or '(not specified)'}\n"
            f"</target_role>"
        )
        resume_block = f"<candidate_resume>\n{self._resume_text}\n</candidate_resume>"
        personal_block = ""
        if self.cfg.personal_context.strip():
            personal_block = (
                f"\n\n<personal_context>\n"
                f"Information about the candidate that is NOT in the resume. Use this "
                f"to answer questions about salary, start date, hobbies, work style, "
                f"availability, location, etc. Treat these as the candidate's actual "
                f"preferences — do not contradict them.\n\n"
                f"{self.cfg.personal_context.strip()}\n"
                f"</personal_context>"
            )

        return [
            {"type": "text", "text": SYSTEM_RULES},
            {
                "type": "text",
                "text": f"{role_block}\n\n{resume_block}{personal_block}",
                "cache_control": {"type": "ephemeral"},
            },
        ]

    def _build_user_message(self, turns: Sequence[Turn], *, style_hint: str = "", summary: str = "") -> str:
        if not turns:
            return "Provide a brief self-introduction in the candidate's voice based on the resume."

        # Full conversation, oldest first (the controller already trims only
        # what's been folded into `summary`; everything else is sent verbatim).
        lines = []
        for t in turns:
            label = "INTERVIEWER" if t.speaker == "interviewer" else "CANDIDATE"
            lines.append(f"[{label}] {t.text.strip()}")

        transcript = "\n".join(lines)
        hint = f"\n\nStyle override for this answer: {style_hint}." if style_hint else ""
        summary_block = (
            f"Summary of the interview before the excerpt below:\n{summary}\n\n" if summary else ""
        )

        return (
            f"{summary_block}"
            f"Conversation so far:\n\n{transcript}\n\n"
            "Answer the LAST interviewer turn above. If the candidate has already started "
            "speaking in response, continue from where they left off; otherwise produce "
            f"the full answer.{hint}"
        )
