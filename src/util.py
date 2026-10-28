"""Small shared helpers."""
from __future__ import annotations

import hashlib

SEED_MODULUS = 2 ** 31


def stable_seed(*parts) -> int:
    """A deterministic seed from arbitrary values.

    Python's built-in `hash` is randomised per process for strings unless PYTHONHASHSEED
    is pinned, so using it for seeding would make donor orders and subsamples differ
    between runs of the same experiment - a resumable job would then mix two designs in
    one results file. This derives the seed from a cryptographic digest instead, which is
    stable across processes, machines and Python versions.
    """
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % SEED_MODULUS


def stable_hex(*parts, size: int = 16) -> str:
    """Full-width digest for filenames, where a collision silently swaps data.

    `stable_seed` folds to 31 bits, which is fine for seeding an RNG but not for naming
    per-user metric files: two units of the same (model, target) collide with probability
    ~1.3e-3 over this grid, and a collision substitutes another unit's vectors with the
    SAME shape - so the shape guard in the paired bootstrap would not catch it. Filenames
    use the untruncated digest instead. Cannot be retrofitted once files exist.
    """
    payload = "|".join(repr(p) for p in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=size).hexdigest()
