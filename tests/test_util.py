"""Seeds must be stable across processes, or a resumed run mixes designs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from util import stable_seed  # noqa: E402


def test_stable_seed_is_deterministic_in_process():
    assert stable_seed("uk", "Books") == stable_seed("uk", "Books")


def test_stable_seed_distinguishes_arguments():
    assert stable_seed("uk", "Books") != stable_seed("de", "Books")
    assert stable_seed(("uk", "Books")) != stable_seed("uk", "Books")


def test_stable_seed_is_in_range():
    for args in [("a",), ("uk", "Books", 3, 0.25), (1, 2, 3)]:
        assert 0 <= stable_seed(*args) < 2 ** 31


def test_stable_seed_survives_hash_randomisation():
    """The reason this module exists: `hash()` on strings varies per process."""
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from util import stable_seed; print(stable_seed('uk', 'Books'))" % SRC
    )
    outputs = set()
    for seed_env in ("0", "1", "12345"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env={"PYTHONHASHSEED": seed_env, **_base_env()})
        outputs.add(out.stdout.strip())
    assert len(outputs) == 1, f"seed varies across processes: {outputs}"


def _base_env():
    import os
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    return env
