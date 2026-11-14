"""
Stable per-computer ID used to bind a license key to one machine.

Source of the raw ID:
  - Windows: HKLM\\SOFTWARE\\Microsoft\\Cryptography\\MachineGuid (created at
    Windows install; changes only on reinstall).
  - macOS:   IOPlatformUUID from IOKit (hardware-bound).
  - Linux:   /etc/machine-id.

The raw value is never shown or sent anywhere. It is hashed with an
app-specific prefix and shortened to 10 bytes, displayed as 16 base32
characters ("ABCD-EFGH-IJKL-MNOP"). That display form is what the customer
copies and sends when they request a key.

Known limitation: virtual machines cloned from one image share the same
MachineGuid, so they share a machine ID (and a key) until Windows
regenerates it (e.g. after sysprep).
"""
from __future__ import annotations

import base64
import hashlib
import re
import subprocess
import sys
from typing import Optional

MACHINE_ID_BYTES = 10           # 80 bits → exactly 16 base32 chars, no padding
_HASH_PREFIX = b"aetherstack-machine-id-v1|"

_cached: Optional[bytes] = None


def _raw_machine_guid() -> str:
    if sys.platform == "win32":
        import winreg
        # KEY_WOW64_64KEY: a 32-bit Python on 64-bit Windows would otherwise be
        # redirected to the WOW6432Node view, which lacks MachineGuid.
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as k:
            value, _ = winreg.QueryValueEx(k, "MachineGuid")
            return str(value)
    if sys.platform == "darwin":
        out = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        m = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', out)
        if not m:
            raise RuntimeError("IOPlatformUUID not found")
        return m.group(1)
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            with open(path, encoding="ascii") as f:
                value = f.read().strip()
            if value:
                return value
        except OSError:
            continue
    raise RuntimeError("no machine-id file found")


def machine_id_bytes() -> bytes:
    """The 10-byte machine ID embedded in license keys. Raises RuntimeError if
    the OS doesn't expose a machine identifier."""
    global _cached
    if _cached is None:
        raw = _raw_machine_guid().strip().lower()
        if not raw:
            raise RuntimeError("empty machine identifier")
        _cached = hashlib.sha256(_HASH_PREFIX + raw.encode("utf-8")).digest()[:MACHINE_ID_BYTES]
    return _cached


def format_machine_id(mid: bytes) -> str:
    """10 bytes → 'ABCD-EFGH-IJKL-MNOP'."""
    s = base64.b32encode(mid).decode("ascii")
    return "-".join(s[i:i + 4] for i in range(0, len(s), 4))


def parse_machine_id(text: str) -> bytes:
    """'abcd-efgh ijkl mnop' (any case, separators ignored) → 10 bytes.
    Raises ValueError if it isn't a well-formed machine ID."""
    s = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if len(s) != 16:
        raise ValueError("a machine ID has 16 letters/digits, e.g. ABCD-EFGH-IJKL-MNOP")
    try:
        out = base64.b32decode(s)
    except Exception as e:
        raise ValueError(f"not a valid machine ID ({e})") from None
    if len(out) != MACHINE_ID_BYTES:
        raise ValueError("not a valid machine ID")
    return out


def machine_id() -> str:
    """Display form of this computer's ID, or "" if it can't be determined."""
    try:
        return format_machine_id(machine_id_bytes())
    except Exception:
        return ""
