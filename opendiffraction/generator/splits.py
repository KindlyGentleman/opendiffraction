"""Group-based dataset splitting ensuring zero object-level data leakage."""

from __future__ import annotations

import hashlib
from typing import Literal


SplitName = Literal["train", "valid", "test", "test_ood"]


def assign_group_to_split(
    group_id: str,
    seed: int = 42,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    is_ood: bool = False,
) -> SplitName:
    """Deterministically assign a group_id to a train/valid/test split.

    All samples sharing group_id (e.g. clean/noisy pairs or different degradations
    of the exact same base object) are guaranteed to reside in the same split.

    If is_ood is True, the group is explicitly assigned to 'test_ood' (evaluated exclusively in testing).
    """
    if is_ood:
        return "test_ood"

    # Deterministic hash mapping
    key = f"{seed}_{group_id}".encode("utf-8")
    hash_val = int(hashlib.sha256(key).hexdigest(), 16) % 10000
    frac = hash_val / 10000.0

    if frac < train_ratio:
        return "train"
    elif frac < (train_ratio + valid_ratio):
        return "valid"
    else:
        return "test"
