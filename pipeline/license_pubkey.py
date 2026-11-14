"""
Public half of the license signing key (Ed25519, raw 32 bytes, base64).

Written by `python tools/license_admin.py init`. It can only VERIFY license
keys; the private half stays with the seller and is never shipped. Replacing
this value invalidates every key signed with the previous private key.
"""
PUBLIC_KEY_B64 = "OvlZwTnSeSwRt9U7UK0mvhGkUuq24qSf1zkTuv7h/2c="
