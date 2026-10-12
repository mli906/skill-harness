"""Deterministic seed derivation (no reliance on Python's salted hash())."""
import hashlib
import random


def derive_seed(seed: int, *parts: object) -> int:
    h = hashlib.sha256(("|".join([str(seed), *map(str, parts)])).encode()).hexdigest()
    return int(h[:15], 16)


def rng(seed: int, *parts: object) -> random.Random:
    return random.Random(derive_seed(seed, *parts))


def task_id(task: str) -> str:
    return "t_" + hashlib.sha1(task.encode()).hexdigest()[:10]
