"""Subject-file subset selection for verbose / graphical audit output.

Both the notebook (EEG snippet plots) and the CLI (annotation printout)
need to pick "a few representative files" from a subject when
``--all-files`` is not appropriate. The selection always includes the
first and last recording so a reviewer can see the endpoints; the
middle picks are randomized with a fixed seed for reproducibility.
"""

from __future__ import annotations

import random
from typing import Sequence, TypeVar


T = TypeVar("T")


def select_files(paths: Sequence[T],
                 *,
                 n_files: int | None = None,
                 seed: int = 0) -> list[T]:
    """Return an ordered subset of ``paths``.

    - ``n_files`` is ``None`` or ``>= len(paths)`` → all paths.
    - ``n_files == 1`` → ``[first]``.
    - ``n_files == 2`` → ``[first, last]``.
    - ``n_files >= 3`` → ``[first, ...random middle..., last]``, with
      middle picks drawn from ``paths[1:-1]`` using ``random.Random(seed)``.

    The result preserves input order (indices are sorted before slicing
    ``paths``), so downstream consumers can iterate in time order.
    """
    n = len(paths)
    if n == 0:
        return []
    if n_files is None or n_files >= n:
        return list(paths)
    if n_files <= 0:
        return []
    if n_files == 1:
        return [paths[0]]
    if n_files == 2:
        return [paths[0], paths[-1]]

    rng = random.Random(seed)
    middle_pool = list(range(1, n - 1))
    picks = rng.sample(middle_pool, k=n_files - 2)
    kept = sorted({0, n - 1, *picks})
    return [paths[i] for i in kept]
