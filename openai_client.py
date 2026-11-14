"""
OpenAI client — same interface as ClaudeClient, streams interview answers
using the candidate's resume as the system prompt.

Uses the official `openai` SDK natively. OpenAI's automatic prompt caching
kicks in for long system prompts (the resume), so we keep the resume at the
top of the system message and the rolling transcript in the user turn.

Model notes (GPT-6.1 Terra live / GPT-6.1 Sol deep, the defaults):
  - Both are reasoning models. `reasoning_effort` controls how much they think
    and therefore time-to-first-token: live answers use cfg.openai_effort
    ("none" = no reasoning, fastest), the Deeper button cfg.openai_deep_effort.
    Set an effort to "" to omit the parameter for a non-reasoning model.
  - Reasoning tokens count toward `max_completion_tokens`; the answer's length
    is shaped by the system prompt, not by that cap.
  - No `temperature`: reasoning models reject custom sampling values.
"""
from __future__ import annotations

import json
from typing import Iterator, Sequence

from openai import OpenAI

from config import Config
from pipeline.answer_prompt import SYSTEM_RULES
from pipeline.context_summary import build_update_prompt
from pipeline.evaluation import (
    EVALUATION_SCHEMA, EVALUATION_USER_PROMPT, InterviewEvaluation,
    format_transcript,
)
from pipeline.types import Turn


def _effort_kw(effort: str) -> dict:
    """`reasoning_effort` kwarg, or nothing when effort is blank (for a
    non-reasoning model, which would reject the parameter)."""
    effort = (effort or "").strip()
    return {"reasoning_effort": effort} if effort else {}


class OpenAIClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not cfg.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Paste it in the API key dialog on launch, "
                "or set the OPENAI_API_KEY environment variable."
            )
        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._resume_text: str = ""

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

    def evaluate_interview(self, turns: Sequence[Turn]) -> InterviewEvaluation:
        """Same shape as ClaudeClient.evaluate_interview — uses OpenAI's
        json_schema response_format."""
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
            # Higher effort for the assessment unless reasoning is turned off
            # for the deep model (blank effort = non-reasoning model).
            eval_effort = "high" if self.cfg.openai_deep_effort.strip() else ""
            response = self._client.chat.completions.create(
                model=self.cfg.openai_deep_model,
                max_completion_tokens=16000,
                **_effort_kw(eval_effort),
                messages=[
                    {"role": "system", "content": self._build_system()},
                    {"role": "user", "content": EVALUATION_USER_PROMPT.format(transcript=transcript)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "interview_evaluation",
                        "schema": EVALUATION_SCHEMA,
                        "strict": True,
                    },
                },
            )
            text = response.choices[0].message.content or "{}"
            data = json.loads(text)
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

    def warmup(self) -> None:
        """
        Tiny one-shot call to establish the HTTPS connection and prime
        OpenAI's automatic prompt cache (which kicks in for prompts ≥1024
        tokens). Same model + effort as live answers so the warmed prefix is
        the one they reuse.
        """
        if not self._resume_text:
            return
        try:
            self._client.chat.completions.create(
                model=self.cfg.openai_model,
                max_completion_tokens=16,
                **_effort_kw(self.cfg.openai_effort),
                messages=[
                    {"role": "system", "content": self._build_system()},
                    {"role": "user", "content": "Ready?"},
                ],
            )
        except Exception:
            pass

    # ── running context summary (see pipeline/context_summary.py) ─────────────
    def summarize(self, prior_summary: str, new_turns: Sequence[Turn]) -> str:
        """Same contract as ClaudeClient.summarize() — see there for why."""
        response = self._client.chat.completions.create(
            model=self.cfg.openai_model,
            max_completion_tokens=2000,
            **_effort_kw(self.cfg.openai_effort),
            messages=[{"role": "user", "content": build_update_prompt(prior_summary, new_turns)}],
        )
        return (response.choices[0].message.content or "").strip()

    def stream_answer(
        self,
        turns: Sequence[Turn],
        *,
        deep: bool = False,
        style_hint: str = "",
        summary: str = "",
    ) -> Iterator[str]:
        if not self._resume_text:
            raise RuntimeError("Call set_context() with the resume before requesting an answer.")

        system = self._build_system()
        user = self._build_user(turns, style_hint=style_hint, summary=summary)
        model = self.cfg.openai_deep_model if deep else self.cfg.openai_model
        effort = self.cfg.openai_deep_effort if deep else self.cfg.openai_effort

        stream = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
            max_completion_tokens=self.cfg.max_tokens,
            **_effort_kw(effort),
        )
        for event in stream:
            try:
                delta = event.choices[0].delta.content
            except (AttributeError, IndexError):
                delta = None
            if delta:
                yield delta

    # ── internals ───────────────────────────────────────────────────────
    def _build_system(self) -> str:
        role_block = (
            f"<target_role>\n{self.cfg.job_title or '(not specified)'}\n\n"
            f"Job description:\n{self.cfg.job_description or '(not specified)'}\n"
            f"</target_role>"
        )
        resume_block = f"<candidate_resume>\n{self._resume_text}\n</candidate_resume>"
        personal_block = ""
        if self.cfg.personal_context.strip():
            personal_block = (
                "\n\n<personal_context>\n"
                "Information about the candidate that is NOT in the resume. Use this "
                "to answer questions about salary, start date, hobbies, work style, "
                "availability, location, etc. Treat these as the candidate's actual "
                "preferences — do not contradict them.\n\n"
                f"{self.cfg.personal_context.strip()}\n"
                "</personal_context>"
            )
        return f"{SYSTEM_RULES}\n\n{role_block}\n\n{resume_block}{personal_block}"

    def _build_user(self, turns: Sequence[Turn], *, style_hint: str = "", summary: str = "") -> str:
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
