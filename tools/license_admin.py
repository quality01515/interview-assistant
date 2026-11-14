"""
License administration — the SELLER's tool. Not imported by the app, so
PyInstaller never bundles it, and the private key it uses never leaves your
computer.

    # once: create the signing key pair and write the public half into the app
    python tools/license_admin.py init

    # issue a key for a customer's machine ID (they copy it from the License dialog)
    python tools/license_admin.py issue --machine ABCD-EFGH-IJKL-MNOP --days 30 --name "Jane Doe"
    python tools/license_admin.py issue --machine ABCD-EFGH-IJKL-MNOP --until 2027-06-30

    # check what a key contains
    python tools/license_admin.py inspect AS2-...

    # this computer's machine ID (to issue yourself a key)
    python tools/license_admin.py machine-id

The private key lives at ~/.aetherstack_license/signing_key.pem by default
(override with --key-file or AETHERSTACK_SIGNING_KEY). BACK IT UP: without it
you can't issue keys that existing builds accept, and anyone who gets it can.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from pipeline import license as lic  # noqa: E402
from pipeline.machine_id import format_machine_id, machine_id, parse_machine_id  # noqa: E402

DEFAULT_KEY_FILE = Path(
    os.environ.get("AETHERSTACK_SIGNING_KEY")
    or Path.home() / ".aetherstack_license" / "signing_key.pem"
)
PUBKEY_FILE = REPO_ROOT / "pipeline" / "license_pubkey.py"


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    if not path.exists():
        sys.exit(f"Signing key not found at {path}. Run `init` first, or pass --key-file.")
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        sys.exit(f"{path} is not an Ed25519 private key.")
    return key


def _public_b64(key: Ed25519PrivateKey) -> str:
    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def _write_pubkey_file(b64: str) -> None:
    text = PUBKEY_FILE.read_text(encoding="utf-8")
    new, n = re.subn(r'^PUBLIC_KEY_B64 = ".*"$', f'PUBLIC_KEY_B64 = "{b64}"', text, flags=re.M)
    if n != 1:
        sys.exit(f"Could not find the PUBLIC_KEY_B64 line in {PUBKEY_FILE}.")
    with open(PUBKEY_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)


def cmd_init(args) -> None:
    path: Path = args.key_file
    if path.exists():
        key = _load_private_key(path)
        print(f"Using the existing signing key at {path}.")
    else:
        key = Ed25519PrivateKey.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        # O_EXCL: never clobber a key that appeared meanwhile.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
        print(f"Created a new signing key at {path}")
        print("BACK IT UP somewhere safe and private. Never commit or share it.")
    b64 = _public_b64(key)
    _write_pubkey_file(b64)
    print(f"Wrote the public key into {PUBKEY_FILE.relative_to(REPO_ROOT)}: {b64}")
    print("Rebuild the app so the new public key ships.")


def _expiry_from_args(args) -> float:
    if args.until:
        try:
            day = dt.date.fromisoformat(args.until)
        except ValueError:
            sys.exit("--until must be YYYY-MM-DD")
        # Valid through the end of that day, local time.
        end = dt.datetime.combine(day, dt.time(23, 59, 59))
        return end.timestamp()
    if args.days is None or args.days <= 0:
        sys.exit("Give --days N (N > 0) or --until YYYY-MM-DD.")
    return time.time() + args.days * 86400


def cmd_issue(args) -> None:
    key = _load_private_key(args.key_file)
    try:
        machine = parse_machine_id(args.machine)
    except ValueError as e:
        sys.exit(f"Bad --machine: {e}")
    expires_at = _expiry_from_args(args)
    if expires_at <= time.time():
        sys.exit("That expiry date is already in the past.")
    try:
        license_key = lic.issue_key(key, machine, expires_at, licensee=args.name or "")
    except ValueError as e:
        sys.exit(str(e))
    # Self-check with the public key that ships in the app.
    from pipeline.license_pubkey import PUBLIC_KEY_B64
    if PUBLIC_KEY_B64 != _public_b64(key):
        print("WARNING: this signing key does not match pipeline/license_pubkey.py — "
              "the app will reject this key. Run `init` with the right key file.", file=sys.stderr)
    lic.check_key(license_key, machine, public_key=base64.b64decode(_public_b64(key)))
    print(f"Machine:  {format_machine_id(machine)}")
    if args.name:
        print(f"Licensee: {args.name}")
    print(f"Expires:  {time.strftime('%Y-%m-%d %H:%M', time.localtime(expires_at))}")
    print()
    print(license_key)


def cmd_inspect(args) -> None:
    try:
        info = lic.decode_key(" ".join(args.key))
    except lic.LicenseError:
        sys.exit("Not a valid key for this app's public key (malformed, edited, or signed with another key).")
    print(f"Machine:  {format_machine_id(info.machine)}")
    print(f"Licensee: {info.licensee or '(none)'}")
    print(f"Issued:   {time.strftime('%Y-%m-%d %H:%M', time.localtime(info.issued_at))}")
    status = "EXPIRED" if time.time() >= info.expires_at else "valid"
    print(f"Expires:  {time.strftime('%Y-%m-%d %H:%M', time.localtime(info.expires_at))} ({status})")


def cmd_machine_id(_args) -> None:
    mid = machine_id()
    if not mid:
        sys.exit("Could not read this computer's machine ID.")
    print(mid)


def main() -> None:
    p = argparse.ArgumentParser(description="AetherStack license administration (seller only).")
    p.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE,
                   help=f"private signing key (default: {DEFAULT_KEY_FILE})")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create the signing key (if missing) and write the public key into the app")
    pi = sub.add_parser("issue", help="issue a key for a machine ID")
    pi.add_argument("--machine", required=True, help="customer's machine ID, e.g. ABCD-EFGH-IJKL-MNOP")
    pi.add_argument("--days", type=int, help="valid for N days from now")
    pi.add_argument("--until", help="valid through this date (YYYY-MM-DD)")
    pi.add_argument("--name", help="licensee name shown in the app (optional)")
    pk = sub.add_parser("inspect", help="show what a key contains")
    pk.add_argument("key", nargs="+")
    sub.add_parser("machine-id", help="print this computer's machine ID")
    args = p.parse_args()
    {"init": cmd_init, "issue": cmd_issue, "inspect": cmd_inspect, "machine-id": cmd_machine_id}[args.cmd](args)


if __name__ == "__main__":
    main()
