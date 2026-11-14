"""
Smoke test for the offline components: transcript, question detector, auto labeler.

Run with:  python smoke_test.py
Exit code 0 = all good.
"""
from __future__ import annotations

import sys

from audio.auto_labeler import AutoLabeler
from pipeline.context_summary import ContextSummarizer
from pipeline.echo_filter import is_echo
from pipeline.interview_health import compute_health
from pipeline import license as lic
from pipeline.license import TRIAL_DAYS, days_remaining, trial_expired
from pipeline.machine_id import format_machine_id, machine_id, parse_machine_id
from pipeline.question_detector import QuestionDetector
from pipeline.transcript import Transcript
from pipeline.types import Turn


def test_transcript_basic() -> None:
    t = Transcript()
    a = t.add("interviewer", "Walk me through your background.")
    b = t.add("candidate", "Sure, I'm a backend engineer with 8 years.")
    assert a is not None and b is not None
    snap = t.snapshot()
    assert len(snap) == 2
    assert t.last_interviewer_turn().text.startswith("Walk me")


def test_transcript_ignores_empty() -> None:
    t = Transcript()
    assert t.add("interviewer", "   ") is None
    assert t.snapshot() == []


def test_transcript_stale_partial_does_not_reorder() -> None:
    # A candidate partial that never finalizes (e.g. an interviewer-bleed
    # blip) must not be resurrected out of order when the candidate later
    # really speaks. The real speech should append at the END, after the
    # interviewer turn that arrived in between — not overwrite the stale
    # partial at its old (earlier) position.
    t = Transcript()
    t.update_partial("candidate", "Continuing", ts=1.0)          # stray, never finalizes
    t.commit("interviewer", "Continuing from the safeguards, can you...", ts=2.0)
    t.update_partial("candidate", "So for a follow-up strategy", ts=3.0)   # real speech

    texts = [(s.speaker, s.text) for s in t.snapshot()]
    # Real candidate speech is last, in chronological order.
    assert texts[-1] == ("candidate", "So for a follow-up strategy"), texts
    # The interviewer turn precedes the candidate's real speech.
    intv_idx = texts.index(("interviewer", "Continuing from the safeguards, can you..."))
    cand_idx = len(texts) - 1
    assert intv_idx < cand_idx, texts


def test_transcript_active_partial_still_replaces_in_place() -> None:
    # Normal case: consecutive partials from the same active speaker replace
    # in place (no duplicate lines) as long as nothing else interleaves.
    t = Transcript()
    t.update_partial("candidate", "So for", ts=1.0)
    t.update_partial("candidate", "So for a follow-up", ts=1.5)
    t.commit("candidate", "So for a follow-up strategy.", ts=2.0)
    texts = [(s.speaker, s.text) for s in t.snapshot()]
    assert texts == [("candidate", "So for a follow-up strategy.")], texts


def test_question_detector_question_mark() -> None:
    qd = QuestionDetector()
    turn = Turn(speaker="interviewer", text="What's your strongest skill?", ts=1.0)
    assert qd.should_answer(turn) is True


def test_question_detector_opener() -> None:
    qd = QuestionDetector()
    turn = Turn(speaker="interviewer", text="Tell me about a hard bug you fixed.", ts=1.0)
    assert qd.should_answer(turn) is True


def test_question_detector_statement_no_fire() -> None:
    qd = QuestionDetector()
    turn = Turn(speaker="interviewer", text="Got it, thanks for that context.", ts=1.0)
    assert qd.should_answer(turn) is False


def test_question_detector_skips_candidate() -> None:
    qd = QuestionDetector()
    turn = Turn(speaker="candidate", text="Should I tell you about the team?", ts=1.0)
    assert qd.should_answer(turn) is False


def test_question_detector_dedup() -> None:
    qd = QuestionDetector()
    turn = Turn(speaker="interviewer", text="How did you solve it?", ts=1.0)
    assert qd.should_answer(turn) is True
    assert qd.should_answer(turn) is False  # same ts → already answered


def test_question_detector_force() -> None:
    qd = QuestionDetector()
    turn = Turn(speaker="interviewer", text="some statement", ts=1.0)
    assert qd.should_answer(turn, force=True) is True


def test_question_detector_candidate_speaking_blocks() -> None:
    qd = QuestionDetector()
    turn = Turn(speaker="interviewer", text="What's your favourite editor?", ts=1.0)
    assert qd.should_answer(turn, candidate_speaking=True) is False


def test_question_detector_matches_opener_after_preamble() -> None:
    # Real-world case: Whisper drops the '?' and the second segment starts
    # with "background, how would you design...". The detector must still fire.
    qd = QuestionDetector()
    turn = Turn(
        speaker="interviewer",
        text="background, how would you design a system to maintain low latency "
             "and high reliability during sudden usage spikes",
        ts=1.0,
    )
    assert qd.should_answer(turn) is True


def test_question_detector_silence_after_fires_for_statement() -> None:
    qd = QuestionDetector()
    turn = Turn(speaker="interviewer", text="Got it, thanks.", ts=1.0)
    # Without silence, a plain statement doesn't trigger.
    assert qd.should_answer(turn) is False
    # But the silence safety net does.
    assert qd.should_answer(turn, silence_after=True) is True


def test_auto_labeler_picks_candidate_by_question_rate() -> None:
    al = AutoLabeler(min_utterances_per_cluster=2)
    # spk_0 asks questions → interviewer
    al.observe("spk_0", "Tell me about your last project.", ts=1.0)
    al.observe("spk_0", "Why did you choose that database?", ts=3.0)
    al.observe("spk_0", "What were the tradeoffs?", ts=5.0)
    # spk_1 answers → candidate
    al.observe("spk_1", "Sure. We built a real-time analytics service.", ts=2.0)
    al.observe("spk_1", "We picked ClickHouse for the throughput.", ts=4.0)

    labels = al.labels()
    assert labels.get("spk_0") == "interviewer", labels
    assert labels.get("spk_1") == "candidate", labels


def test_auto_labeler_swap_and_lock_inverts_and_freezes() -> None:
    al = AutoLabeler(min_utterances_per_cluster=2)
    al.observe("spk_0", "Tell me about your last project.", ts=1.0)
    al.observe("spk_0", "Why ClickHouse?", ts=3.0)
    al.observe("spk_1", "We built a real-time analytics service.", ts=2.0)
    al.observe("spk_1", "Throughput was the main reason.", ts=4.0)

    before = al.labels()
    assert before["spk_0"] == "interviewer"
    assert before["spk_1"] == "candidate"

    al.swap_and_lock()
    after = al.labels()
    assert after["spk_0"] == "candidate"
    assert after["spk_1"] == "interviewer"
    assert al.locked is True

    # Even if the model would normally re-flip, the lock keeps the user's pick.
    for _ in range(5):
        al.observe("spk_0", "Why ClickHouse?", ts=10.0)
    assert al.labels() == after


def test_echo_filter_drops_near_duplicate_within_lag_window() -> None:
    # Real acoustic echo: mic captures the speaker output ~0.1s after the
    # interviewer's audio ends. Same words, same time. Should drop.
    interviewer = Turn(
        speaker="interviewer",
        text="background, how would you design a system to maintain low latency "
             "and high reliability during sudden usage spikes such as a pandemic "
             "driven surge while also respecting compliance requirements.",
        ts=110.0,
    )
    echo_text = (
        "Given that background, how would you design a system to maintain low "
        "latency and high reliability during sudden usage spikes."
    )
    assert is_echo(echo_text, [interviewer], candidate_ts=110.5) is True


def test_echo_filter_keeps_candidate_reading_answer_aloud() -> None:
    # The bug the user hit on small.en: candidate reads the suggested answer
    # aloud, which legitimately echoes question words. But it happens SEVERAL
    # seconds after the interviewer finished — well outside the echo window.
    interviewer = Turn(
        speaker="interviewer",
        text="tell me about scaling distributed systems",
        ts=100.0,
    )
    # Candidate started reading the answer 4s after interviewer stopped;
    # finished reading at ts 12s — total lag from interviewer end: 12s.
    candidate_reading = (
        "Sure. When I think about scaling distributed systems, the key "
        "consideration is partitioning and load balancing strategies."
    )
    assert is_echo(candidate_reading, [interviewer], candidate_ts=112.0) is False


def test_echo_filter_keeps_genuine_candidate_speech() -> None:
    interviewer = Turn(
        speaker="interviewer",
        text="Tell me about a hard system you designed.",
        ts=100.0,
    )
    candidate_real = (
        "Sure. The hardest one was a real-time analytics pipeline using ClickHouse and Kafka — "
        "we hit ten million events per second during Black Friday."
    )
    assert is_echo(candidate_real, [interviewer], candidate_ts=102.0) is False


def test_echo_filter_ignores_short_utterances() -> None:
    interviewer = Turn(speaker="interviewer", text="Right, yeah, makes sense.", ts=100.0)
    short = "Yeah, right."
    assert is_echo(short, [interviewer], candidate_ts=100.5) is False


def test_echo_filter_respects_lag_window() -> None:
    # Same words but >3s after the interviewer ended — not echo.
    old = Turn(speaker="interviewer", text="Tell me about your background", ts=100.0)
    same_text = "Tell me about your background you mentioned in the intro"
    assert is_echo(same_text, [old], candidate_ts=110.0) is False


def test_echo_filter_catches_echo_during_in_progress_interviewer_turn() -> None:
    # Long interviewer turn still in progress (not yet committed): its text
    # lives in snapshot() as a partial but NOT in snapshot_finalized(). The
    # mic's acoustic echo of the tail finalizes before the interviewer turn
    # commits. The filter must catch it — which only works when it compares
    # against the full snapshot (the app.py fix), not finalized turns only.
    t = Transcript()
    t.update_partial(
        "interviewer",
        "tell me about a time you had to scale a distributed system under heavy load",
        ts=100.0,
    )
    echo_text = "scale a distributed system under heavy load"
    # Finalized-only (the old behavior) can't see the in-progress turn:
    assert is_echo(echo_text, t.snapshot_finalized(), candidate_ts=100.3) is False
    # Full snapshot (the fix) correctly flags the echo:
    assert is_echo(echo_text, t.snapshot(), candidate_ts=100.3) is True


def test_health_no_turns() -> None:
    h = compute_health([])
    assert h.score == 70 and h.label == "waiting"


def test_health_strong_for_normal_length_answer() -> None:
    turns = [
        Turn(speaker="interviewer", text="Tell me about a system you built.", ts=1.0),
        Turn(
            speaker="candidate",
            text=("Sure. We built a real-time analytics pipeline on ClickHouse, handling "
                  "50 million events per day with sub-100ms p99 query latency. The cost "
                  "was about a third of Snowflake at our scale."),
            ts=2.0,
        ),
    ]
    h = compute_health(turns)
    assert h.score >= 75, h
    assert h.label == "strong"


def test_health_drops_for_one_word_answers() -> None:
    turns = [
        Turn(speaker="interviewer", text="Walk me through your background.", ts=1.0),
        Turn(speaker="candidate", text="Uh, yeah.", ts=2.0),
    ]
    h = compute_health(turns)
    assert h.score < 70, h


# ── license (v2: signed, machine-bound, time-limited) ────────────────────────
# Tests sign with a throwaway key pair and pass its public half explicitly, so
# they never need (or touch) the real signing key.

_DAY = 86400.0
_NOW = 1_800_000_000.0
_MACHINE = bytes(range(10))
_OTHER_MACHINE = bytes(range(1, 11))
_LEGACY_KEY = "A301-CED5-7D6B-7293-222E-C1BA"   # the retired shared v1 key


def _test_keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return priv, pub


def _eval(key, *, first_run_at=_NOW, now=_NOW, pub=None, **kw):
    return lic.evaluate(key, first_run_at=first_run_at, now=now,
                        machine=_MACHINE, public_key=pub, **kw)


def test_license_v2_roundtrip_and_wrapped_key() -> None:
    priv, pub = _test_keypair()
    key = lic.issue_key(priv, _MACHINE, _NOW + 30 * _DAY, licensee="Jane Doe", issued_at=_NOW)
    assert key.startswith("AS2-")
    info = lic.check_key(key, _MACHINE, now=_NOW + _DAY, public_key=pub)
    assert info.licensee == "Jane Doe" and info.expires_at == int(_NOW + 30 * _DAY)
    # E-mail clients wrap long lines — whitespace inside the key is ignored.
    wrapped = "\n  ".join(key[i:i + 40] for i in range(0, len(key), 40))
    assert lic.check_key(wrapped, _MACHINE, now=_NOW, public_key=pub).licensee == "Jane Doe"


def test_license_v2_rejections() -> None:
    priv, pub = _test_keypair()
    key = lic.issue_key(priv, _MACHINE, _NOW + 30 * _DAY, issued_at=_NOW)

    def reason(k, machine=_MACHINE, now=_NOW):
        try:
            lic.check_key(k, machine, now=now, public_key=pub)
        except lic.LicenseError as e:
            return e.reason
        return "ok"

    assert reason(key) == "ok"
    assert reason(key, machine=_OTHER_MACHINE) == lic.R_WRONG_MACHINE
    assert reason(key, now=_NOW + 31 * _DAY) == lic.R_LICENSE_EXPIRED
    assert reason(key, now=_NOW - 2 * _DAY) == lic.R_CLOCK            # clock before issue date
    tampered = key[:-6] + ("A" if key[-6] != "A" else "B") + key[-5:]
    assert reason(tampered) == lic.R_INVALID
    other_priv, _ = _test_keypair()                                    # signed by someone else
    assert reason(lic.issue_key(other_priv, _MACHINE, _NOW + _DAY, issued_at=_NOW)) == lic.R_INVALID
    assert reason("") == lic.R_INVALID and reason("AS2-garbage") == lic.R_INVALID
    assert reason(_LEGACY_KEY) == lic.R_INVALID                        # old key isn't a v2 key


def test_license_evaluate_states() -> None:
    priv, pub = _test_keypair()
    trial_over = _NOW - (TRIAL_DAYS + 5) * _DAY
    key = lic.issue_key(priv, _MACHINE, _NOW + 10 * _DAY, issued_at=_NOW - _DAY)

    st = _eval(key, first_run_at=trial_over, pub=pub)
    assert st.status == lic.S_LICENSED and st.days_left == 10, st
    assert _eval("", pub=pub).status == lic.S_TRIAL
    st = _eval("", first_run_at=trial_over, pub=pub)
    assert st.status == lic.S_BLOCKED and st.reason == lic.R_TRIAL_EXPIRED
    # Expired license after the trial → blocked with the specific reason.
    st = _eval(key, first_run_at=trial_over, now=_NOW + 11 * _DAY, pub=pub)
    assert st.status == lic.S_BLOCKED and st.reason == lic.R_LICENSE_EXPIRED
    # A bad key during the trial keeps the trial running, with the problem noted.
    st = _eval("AS2-nonsense", pub=pub)
    assert st.status == lic.S_TRIAL and st.reason == lic.R_INVALID


def test_license_clock_rollback_blocks() -> None:
    priv, pub = _test_keypair()
    key = lic.issue_key(priv, _MACHINE, _NOW + 10 * _DAY, issued_at=_NOW - _DAY)
    # Within a day of the last-seen time is tolerated (NTP / manual fixes).
    assert _eval(key, last_seen=_NOW + 12 * 3600, pub=pub).status == lic.S_LICENSED
    st = _eval(key, last_seen=_NOW + 3 * _DAY, pub=pub)
    assert st.status == lic.S_BLOCKED and st.reason == lic.R_CLOCK
    # Applies during the trial too.
    assert _eval("", last_seen=_NOW + 3 * _DAY, pub=pub).reason == lic.R_CLOCK


def test_license_legacy_key_grace_period() -> None:
    trial_over = _NOW - (TRIAL_DAYS + 5) * _DAY
    assert lic.is_legacy_key(_LEGACY_KEY) and lic.is_legacy_key(_LEGACY_KEY.lower().replace("-", ""))
    st = _eval(_LEGACY_KEY, first_run_at=trial_over)                  # first launch of this version
    assert st.status == lic.S_LEGACY_GRACE and st.days_left == lic.LEGACY_GRACE_DAYS, st
    st = _eval(_LEGACY_KEY, first_run_at=trial_over, legacy_grace_started_at=_NOW - 5 * _DAY)
    assert st.status == lic.S_LEGACY_GRACE and st.days_left == 2, st
    st = _eval(_LEGACY_KEY, first_run_at=trial_over, legacy_grace_started_at=_NOW - 8 * _DAY)
    assert st.status == lic.S_BLOCKED and st.reason == lic.R_LEGACY_RETIRED, st


def test_machine_id_format_roundtrip() -> None:
    shown = format_machine_id(_MACHINE)
    assert len(shown) == 19 and shown.count("-") == 3, shown
    assert parse_machine_id(shown) == _MACHINE
    assert parse_machine_id(shown.lower().replace("-", " ")) == _MACHINE
    for bad in ("", "ABCD-EFGH", "ABCD-EFGH-IJKL-MNO1"):
        try:
            parse_machine_id(bad)
            raise AssertionError(f"accepted {bad!r}")
        except ValueError:
            pass
    mid = machine_id()                 # this computer: stable across calls
    assert mid and mid == machine_id(), mid


def test_trial_countdown() -> None:
    import time
    now = time.time()
    assert days_remaining(0) == TRIAL_DAYS
    assert days_remaining(now, now=now) == TRIAL_DAYS
    assert days_remaining(now - 5 * 86400, now=now) == TRIAL_DAYS - 5
    assert days_remaining(now - TRIAL_DAYS * 86400, now=now) == 0
    assert days_remaining(now - 100 * 86400, now=now) == 0
    assert not trial_expired(now - 5 * 86400, now=now)
    assert trial_expired(now - (TRIAL_DAYS + 1) * 86400, now=now)


def _turns(n: int, start_ts: float = 1.0) -> list[Turn]:
    out = []
    for i in range(n):
        speaker = "interviewer" if i % 2 == 0 else "candidate"
        out.append(Turn(speaker=speaker, text=f"turn {i}", ts=start_ts + i))
    return out


def test_context_summary_under_budget_folds_nothing() -> None:
    cs = ContextSummarizer()
    turns = _turns(5)
    # Whole transcript fits the budget → nothing folded, everything verbatim.
    assert cs.turns_to_fold(turns, budget_tokens=10_000, min_verbatim=12) == []
    assert cs.should_update(turns, 10_000, 12) is False
    assert [t.text for t in cs.unfolded(turns)] == [t.text for t in turns]


def test_context_summary_folds_oldest_over_budget() -> None:
    cs = ContextSummarizer()
    turns = _turns(20)
    # budget=0 forces folding down to the min-verbatim floor: 20 - 12 = 8 oldest.
    to_fold = cs.turns_to_fold(turns, budget_tokens=0, min_verbatim=12)
    assert [t.text for t in to_fold] == [f"turn {i}" for i in range(8)]
    assert cs.should_update(turns, 0, 12) is True


def test_context_summary_never_folds_below_min_verbatim() -> None:
    cs = ContextSummarizer()
    turns = _turns(14)
    # Even wildly over budget, keep >= min_verbatim recent turns verbatim.
    to_fold = cs.turns_to_fold(turns, budget_tokens=0, min_verbatim=12)
    assert len(to_fold) == 2                       # 14 - 12
    # With only min_verbatim turns, nothing is foldable.
    assert cs.turns_to_fold(_turns(12), budget_tokens=0, min_verbatim=12) == []


def test_context_summary_respects_token_budget() -> None:
    cs = ContextSummarizer()
    # 10 turns of 40 chars each = 400 chars => 100 est-tokens total.
    turns = [Turn(speaker="candidate", text="x" * 40, ts=1.0 + i) for i in range(10)]
    # budget 50 tokens (=200 chars => 5 turns), min_verbatim small so budget binds.
    to_fold = cs.turns_to_fold(turns, budget_tokens=50, min_verbatim=2)
    assert len(to_fold) == 5                       # keep 5 (=50 tokens) verbatim


def test_context_summary_apply_update_advances_cursor() -> None:
    cs = ContextSummarizer()
    turns = _turns(20)
    to_fold = cs.turns_to_fold(turns, budget_tokens=0, min_verbatim=12)
    cs.apply_update(to_fold, "The candidate discussed X and Y.")
    assert cs.summary == "The candidate discussed X and Y."
    # Folded turns drop out of the verbatim set exactly as they enter the summary.
    assert [t.text for t in cs.unfolded(turns)] == [f"turn {i}" for i in range(8, 20)]
    # Only the min-verbatim floor remains → nothing more to fold.
    assert cs.turns_to_fold(turns, budget_tokens=0, min_verbatim=12) == []


def test_context_summary_folded_turns_dont_repeat_on_growth() -> None:
    cs = ContextSummarizer()
    turns = _turns(20)
    cs.apply_update(cs.turns_to_fold(turns, 0, 12), "summary so far")
    # More turns arrive; already-folded turns must not reappear as verbatim.
    grown = _turns(30)
    assert [t.text for t in cs.unfolded(grown)] == [f"turn {i}" for i in range(8, 30)]
    # And the next fold takes the oldest of the NEW unfolded set, never re-folds.
    again = cs.turns_to_fold(grown, 0, 12)
    assert again[0].text == "turn 8"


def test_context_summary_reset_clears_state() -> None:
    cs = ContextSummarizer()
    turns = _turns(20)
    cs.apply_update(cs.turns_to_fold(turns, 0, 12), "summary")
    cs.reset()
    assert cs.summary == ""
    # Cursor reset → the whole transcript is verbatim again.
    assert [t.text for t in cs.unfolded(turns)] == [t.text for t in turns]


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  OK    {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
