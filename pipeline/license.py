"""
Trial + license management.

Trial
    A 30-day trial starts on first launch (`first_run_at` in config.json).

License keys (v2)
    A key is a small payload — machine ID, issue date, expiry date, licensee
    name — signed with an Ed25519 PRIVATE key that only the seller holds
    (tools/license_admin.py; never shipped). The app embeds only the matching
    PUBLIC key (pipeline/license_pubkey.py), which can verify a key but cannot
    create one, so extracting it from the .exe doesn't let anyone mint keys.
    A key is valid only on the machine it was issued for and only until its
    expiry date.

    Format:  "AS2-" + base64url( payload || signature )
      payload = b"AS" | version (1 byte = 2) | machine ID (10 bytes)
                | issued_at (u32, unix seconds) | expires_at (u32) | licensee (UTF-8)
      signature = Ed25519 over the payload (64 bytes)

Clock rollback
    The latest time the app has seen is stored (`license_last_seen`). If the
    clock is earlier than that by more than CLOCK_TOLERANCE_SEC, or earlier
    than a key's issue date, the app refuses to run until the clock is fixed.

Retired v1 key
    The old shared key (one HMAC key for every copy) is no longer accepted for
    activation. Installs that already had it get LEGACY_GRACE_DAYS from their
    first launch of this version to switch to a v2 key.

Limits: this is an offline check. Someone who modifies the app's code can
skip it; the goal is to stop key sharing and casual copying.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import math
import re
import struct
import time
from dataclasses import dataclass
from typing import Optional

TRIAL_DAYS = 30
LEGACY_GRACE_DAYS = 7
CLOCK_TOLERANCE_SEC = 24 * 3600     # NTP corrections / manual fixes inside a day are fine

KEY_PREFIX = "AS2-"
_MAGIC = b"AS"
_VERSION = 2
_FIXED = struct.Struct(">2sB10sII")  # magic, version, machine, issued_at, expires_at
_SIG_LEN = 64
MAX_LICENSEE_BYTES = 60

# Blocked / problem reasons
R_TRIAL_EXPIRED = "trial_expired"
R_LICENSE_EXPIRED = "license_expired"
R_WRONG_MACHINE = "wrong_machine"
R_INVALID = "invalid"
R_LEGACY_RETIRED = "legacy_retired"
R_CLOCK = "clock"
R_NO_MACHINE_ID = "no_machine_id"

# States
S_LICENSED = "licensed"
S_TRIAL = "trial"
S_LEGACY_GRACE = "legacy_grace"
S_BLOCKED = "blocked"


class LicenseError(Exception):
    def __init__(self, reason: str, info: "Optional[LicenseInfo]" = None):
        super().__init__(reason)
        self.reason = reason
        self.info = info


@dataclass
class LicenseInfo:
    machine: bytes
    issued_at: float
    expires_at: float
    licensee: str


@dataclass
class LicenseState:
    status: str                       # S_LICENSED | S_TRIAL | S_LEGACY_GRACE | S_BLOCKED
    days_left: int = 0
    reason: str = ""                  # why blocked, or the problem with a stored key
    info: Optional[LicenseInfo] = None
    last_seen: float = 0.0            # for R_CLOCK messages

    @property
    def ok(self) -> bool:
        return self.status != S_BLOCKED


# ── trial accounting ─────────────────────────────────────────────────────────

def days_remaining(first_run_at: float, now: float | None = None) -> int:
    """How many trial days are left. Capped at 0."""
    if first_run_at <= 0:
        return TRIAL_DAYS
    now = now if now is not None else time.time()
    elapsed_days = (now - first_run_at) / 86400.0
    return max(0, TRIAL_DAYS - int(elapsed_days))


def trial_expired(first_run_at: float, now: float | None = None) -> bool:
    return days_remaining(first_run_at, now) <= 0


def _days_until(t: float, now: float) -> int:
    return max(0, math.ceil((t - now) / 86400.0))


# ── retired v1 key (recognised only to grant the grace period) ───────────────

_LEGACY_SECRET = b"aetherstack-interview-assistant-v1-license-secret"
_LEGACY_PAYLOAD = b"aetherstack-interview-assistant"


def is_legacy_key(key: str) -> bool:
    norm = re.sub(r"[^A-Za-z0-9]", "", key or "").upper()
    expected = hmac.new(_LEGACY_SECRET, _LEGACY_PAYLOAD, hashlib.sha256).hexdigest()[:24].upper()
    return len(norm) == len(expected) and hmac.compare_digest(norm, expected)


# ── v2 keys ──────────────────────────────────────────────────────────────────

def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _load_public_key(public_key: bytes | None):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if public_key is None:
        from pipeline.license_pubkey import PUBLIC_KEY_B64
        if not PUBLIC_KEY_B64:
            raise LicenseError(R_INVALID)
        public_key = base64.b64decode(PUBLIC_KEY_B64)
    return Ed25519PublicKey.from_public_bytes(public_key)


def looks_like_v2_key(key: str) -> bool:
    return re.sub(r"\s+", "", key or "").upper().startswith(KEY_PREFIX)


def decode_key(key: str, public_key: bytes | None = None) -> LicenseInfo:
    """Check the signature and decode the payload. Does NOT check machine or
    expiry. Raises LicenseError(R_INVALID) for anything malformed or forged."""
    from cryptography.exceptions import InvalidSignature

    compact = re.sub(r"\s+", "", key or "")   # e-mail clients wrap long lines
    if not compact.upper().startswith(KEY_PREFIX):
        raise LicenseError(R_INVALID)
    try:
        blob = _b64url_decode(compact[len(KEY_PREFIX):])
    except Exception:
        raise LicenseError(R_INVALID) from None
    if len(blob) < _FIXED.size + _SIG_LEN:
        raise LicenseError(R_INVALID)
    payload, sig = blob[:-_SIG_LEN], blob[-_SIG_LEN:]
    try:
        _load_public_key(public_key).verify(sig, payload)
    except InvalidSignature:
        raise LicenseError(R_INVALID) from None
    magic, version, machine, issued, expires = _FIXED.unpack(payload[:_FIXED.size])
    if magic != _MAGIC or version != _VERSION:
        raise LicenseError(R_INVALID)
    try:
        licensee = payload[_FIXED.size:].decode("utf-8")
    except UnicodeDecodeError:
        raise LicenseError(R_INVALID) from None
    return LicenseInfo(machine=machine, issued_at=float(issued),
                       expires_at=float(expires), licensee=licensee)


def check_key(
    key: str,
    machine: bytes,
    now: float | None = None,
    public_key: bytes | None = None,
) -> LicenseInfo:
    """Full v2 check for THIS machine at `now`. Returns the decoded info, or
    raises LicenseError with the reason (and the decoded info when the
    signature itself was fine)."""
    now = time.time() if now is None else now
    info = decode_key(key, public_key)
    if not hmac.compare_digest(info.machine, machine):
        raise LicenseError(R_WRONG_MACHINE, info)
    if now < info.issued_at - CLOCK_TOLERANCE_SEC:
        raise LicenseError(R_CLOCK, info)
    if now >= info.expires_at:
        raise LicenseError(R_LICENSE_EXPIRED, info)
    return info


def issue_key(
    private_key,
    machine: bytes,
    expires_at: float,
    licensee: str = "",
    issued_at: float | None = None,
) -> str:
    """Create a v2 key. `private_key` is a cryptography Ed25519PrivateKey —
    only the seller's tool (tools/license_admin.py) and tests call this."""
    issued_at = time.time() if issued_at is None else issued_at
    name = licensee.strip().encode("utf-8")
    if len(name) > MAX_LICENSEE_BYTES:
        raise ValueError(f"licensee name is limited to {MAX_LICENSEE_BYTES} bytes")
    if len(machine) != 10:
        raise ValueError("machine ID must be 10 bytes")
    if expires_at <= issued_at:
        raise ValueError("expiry must be after the issue date")
    payload = _FIXED.pack(_MAGIC, _VERSION, machine, int(issued_at), int(expires_at)) + name
    sig = private_key.sign(payload)
    return KEY_PREFIX + base64.urlsafe_b64encode(payload + sig).decode("ascii").rstrip("=")


# ── overall state ────────────────────────────────────────────────────────────

def evaluate(
    license_key: str,
    *,
    first_run_at: float,
    last_seen: float = 0.0,
    legacy_grace_started_at: float = 0.0,
    now: float | None = None,
    machine: bytes | None = None,
    public_key: bytes | None = None,
) -> LicenseState:
    """
    Decide whether the app may run. Pure function (no I/O except reading the
    machine ID when `machine` is None) — the caller persists `first_run_at`,
    `license_last_seen` and `legacy_grace_started_at`.

    Order: clock rollback → valid v2 key → old key's grace period → trial.
    A stored key with a problem (expired, other machine) still lets the trial
    run if the trial has days left; otherwise its problem is the reason shown.
    """
    now = time.time() if now is None else now
    if last_seen > 0 and now < last_seen - CLOCK_TOLERANCE_SEC:
        return LicenseState(S_BLOCKED, reason=R_CLOCK, last_seen=last_seen)

    key = (license_key or "").strip()
    key_problem = ""
    key_info: Optional[LicenseInfo] = None

    if key and looks_like_v2_key(key):
        if machine is None:
            try:
                from pipeline.machine_id import machine_id_bytes
                machine = machine_id_bytes()
            except Exception:
                machine = None
        if machine is None:
            key_problem = R_NO_MACHINE_ID
        else:
            try:
                info = check_key(key, machine, now, public_key)
                return LicenseState(S_LICENSED, days_left=_days_until(info.expires_at, now), info=info)
            except LicenseError as e:
                if e.reason == R_CLOCK:
                    return LicenseState(S_BLOCKED, reason=R_CLOCK, info=e.info,
                                        last_seen=e.info.issued_at if e.info else 0.0)
                key_problem, key_info = e.reason, e.info
    elif key and is_legacy_key(key):
        start = legacy_grace_started_at if legacy_grace_started_at > 0 else now
        grace_end = start + LEGACY_GRACE_DAYS * 86400
        if now < grace_end:
            return LicenseState(S_LEGACY_GRACE, days_left=_days_until(grace_end, now))
        key_problem = R_LEGACY_RETIRED
    elif key:
        key_problem = R_INVALID

    trial_days = days_remaining(first_run_at, now)
    if trial_days > 0:
        return LicenseState(S_TRIAL, days_left=trial_days, reason=key_problem, info=key_info)
    return LicenseState(S_BLOCKED, reason=key_problem or R_TRIAL_EXPIRED, info=key_info)


# ── user-facing text ─────────────────────────────────────────────────────────

def _date(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(t))


def describe(reason: str, info: Optional[LicenseInfo] = None, last_seen: float = 0.0) -> str:
    """One or two sentences explaining a problem reason to the user."""
    if reason == R_TRIAL_EXPIRED:
        return f"Your {TRIAL_DAYS}-day trial has ended. Enter a license key to keep using the app."
    if reason == R_LICENSE_EXPIRED:
        when = f" on {_date(info.expires_at)}" if info else ""
        return f"This license expired{when}. Send your machine ID to get a renewed key."
    if reason == R_WRONG_MACHINE:
        return ("This license key was issued for a different computer. "
                "Keys only work on the machine whose ID they were made for.")
    if reason == R_LEGACY_RETIRED:
        return ("The old shared license key has been retired. "
                "Send your machine ID to get a personal key for this computer.")
    if reason == R_CLOCK:
        when = f" ({_date(last_seen)})" if last_seen else ""
        return ("Your computer's date is earlier than the last time this app ran or "
                f"than the license's issue date{when}. Correct the date and time, then click Retry.")
    if reason == R_NO_MACHINE_ID:
        return "This computer's machine ID could not be read, so a license key can't be checked."
    if reason == R_INVALID:
        return "That license key is not valid. Check that you copied all of it."
    return ""


def status_line(state: LicenseState) -> str:
    """Short status for the setup screen."""
    s = "s" if state.days_left != 1 else ""
    if state.status == S_LICENSED:
        who = f"Licensed to {state.info.licensee}" if state.info and state.info.licensee else "Licensed"
        return f"{who}  ·  expires {_date(state.info.expires_at)} ({state.days_left} day{s} left)"
    if state.status == S_LEGACY_GRACE:
        return (f"Your old license key stops working in {state.days_left} day{s}. "
                "Click License… and send your machine ID to get a new key.")
    if state.status == S_TRIAL:
        line = f"Trial: {state.days_left} day{s} left"
        if state.reason:
            line += f"  ·  {describe(state.reason, state.info)}"
        return line
    return describe(state.reason, state.info, state.last_seen)
