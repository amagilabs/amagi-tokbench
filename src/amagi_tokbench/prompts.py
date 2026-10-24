"""The fixed prompt set.

Every run of every backend gets the same three cases, so two cards made with the same
prompt set version can be compared. The long texts are built from a seeded generator, not
read from a file: same seed, same words, on any machine. Change anything here and bump
PROMPT_SET_VERSION (a test pins the hash).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

PROMPT_SET_VERSION = "v1"

_SUBJECTS = [
    "the north pump", "the cold room", "the loading dock", "the spare router", "the east fence",
    "the night shift", "the delivery van", "the back office printer", "the rain gauge",
    "the solar inverter", "the walk-in freezer", "the label printer", "the water heater",
    "the forklift", "the main switch", "the greenhouse vent", "the security camera",
    "the coffee machine", "the parts shelf", "the tool crib",
]
_EVENTS = [
    "was checked and logged as normal", "made a grinding noise for about ten minutes",
    "was moved two meters to the left", "needed a new filter", "stopped twice before noon",
    "was cleaned and put back in service", "showed a warning light that cleared on restart",
    "was borrowed by the other crew", "ran warm but stayed inside its limits",
    "got a firmware update", "was tagged for repair", "passed the weekly inspection",
    "lost power during the storm", "was counted and came up three short",
    "had its schedule changed to mornings",
]
_FOLLOWUPS = [
    "Someone should look at it again on Friday.", "No further action is planned.",
    "The supplier was called and promised a callback.", "A photo was added to the shared folder.",
    "The fix held for the rest of the day.", "We ordered a replacement part.",
    "The manager signed off on the change.", "It will be watched for another week.",
    "Nobody could say when it started.", "The manual suggests doing this every month.",
]
_OPENERS = ["At", "Around", "Just after", "A little before"]


def _log_text(seed: int, entries: int) -> str:
    rng = random.Random(seed)
    lines = []
    for i in range(1, entries + 1):
        hour = 6 + (i * 7) % 14
        minute = (i * 13) % 60
        lines.append(
            f"Entry {i}. {rng.choice(_OPENERS)} {hour:02d}:{minute:02d}, "
            f"{rng.choice(_SUBJECTS)} {rng.choice(_EVENTS)}. {rng.choice(_FOLLOWUPS)}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    prompt: str
    max_tokens: int

    @property
    def words(self) -> int:
        return len(self.prompt.split())


_ASK_LONG = (
    "Below is a maintenance log from a small warehouse. Write a detailed report for the owner: "
    "group the entries by piece of equipment, say what went wrong with each one, and list what "
    "should happen next. Use full sentences.\n\n"
)

CASES = (
    Case(
        name="short",
        description="short chat prompt, long answer: generation speed",
        prompt=(
            "Explain in plain words how a refrigerator keeps food cold. Cover the compressor, the "
            "refrigerant, the coils and the thermostat, one paragraph each."
        ),
        max_tokens=256,
    ),
    Case(
        name="medium",
        description="about 600 prompt tokens: everyday chat with context",
        prompt=_ASK_LONG + _log_text(seed=618, entries=18),
        max_tokens=128,
    ),
    Case(
        name="long",
        description="about 2,500 prompt tokens: prompt processing speed",
        prompt=_ASK_LONG + _log_text(seed=1066, entries=74),
        max_tokens=128,
    ),
)

CASE_NAMES = tuple(c.name for c in CASES)


def get_cases(names=None):
    """Return the cases to run, in the fixed order, optionally filtered by name."""
    if not names:
        return list(CASES)
    unknown = [n for n in names if n not in CASE_NAMES]
    if unknown:
        raise ValueError(f"unknown case {', '.join(unknown)} (choose from {', '.join(CASE_NAMES)})")
    return [c for c in CASES if c.name in names]


def prompt_set_hash() -> str:
    h = hashlib.sha256()
    for c in CASES:
        h.update(f"{c.name}\0{c.max_tokens}\0{c.prompt}\0".encode("utf-8"))
    return h.hexdigest()


def with_nonce(prompt: str, nonce: str) -> str:
    """Put a unique tag at the very start of the prompt.

    Servers reuse cached prompt prefixes (llama.cpp, vLLM and Ollama all do), which would make
    repeat runs look like instant prompt processing. A different first line per request keeps
    every run honest without changing the prompt length by more than a few tokens.
    """
    return f"[run {nonce}]\n{prompt}"
