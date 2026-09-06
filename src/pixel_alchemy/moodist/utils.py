"""Port of `moodist/src/helpers/random.ts`."""

from __future__ import annotations

import random as _random


def random(min: float, max: float) -> float:
    return _random.random() * (max - min) + min


def random_int(min: int, max: int) -> int:
    return int(random(min, max))


def pick(array: list, **_kw):
    if not array:
        raise ValueError("array shouldn't be empty")
    return _random.choice(array)


def pick_many(array: list, count: int) -> list:
    return _random.sample(array, min(count, len(array)))


def shuffle(array: list) -> list:
    arr = array[:]
    _random.shuffle(arr)
    return arr
