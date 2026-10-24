"""Paired contrasts of probe scores: seed and held-out-episode uncertainty, and the MDD.

A contrast ``a - b`` compares two families of encoders (seeds) scored on the **same** held-out
windows. Two sources of uncertainty are reported separately and then combined:

* seed: training noise on these episodes (paired seed differences when ``a`` and ``b`` share
  seeds, otherwise the two families' variances);
* episode: which held-out episodes were drawn, from a bootstrap that resamples the same episodes
  for every encoder of both families (``rng(0)``, as in ``scripts/controls_test.py``).
"""

from dataclasses import dataclass

import numpy as np

from sensorium.model.probes import Predictions

DRAWS = 300
MDD_FACTOR = 2.8
"""80% power at 5% two-sided: the minimum detectable difference is 2.8 standard errors."""


@dataclass(frozen=True)
class Contrast:
    """``a - b`` of the probe mean.

    Attributes:
        point: Difference of the family means (the mean of the same-seed differences if paired).
        seed_se: Standard error of ``point`` from training seeds.
        episode_se: Standard deviation of the difference over bootstrap resamples of the held-out
            episodes (one resample shared by every encoder).
        n_a, n_b: Encoders in each family.
    """

    point: float
    seed_se: float
    episode_se: float
    n_a: int
    n_b: int

    @property
    def se(self) -> float:
        """Combined standard error (the two components in quadrature)."""
        return float(np.hypot(self.seed_se, self.episode_se))

    @property
    def mdd(self) -> float:
        """Minimum detectable difference: ``MDD_FACTOR`` x the combined SE."""
        return MDD_FACTOR * self.se

    @property
    def resolved(self) -> bool:
        """Whether ``|point|`` exceeds the MDD."""
        return bool(abs(self.point) > self.mdd)


_BOOTS: dict[tuple[int, int], tuple[Predictions, np.ndarray]] = {}


def boot_means(pred: Predictions, draws: int = DRAWS) -> np.ndarray:
    """Probe mean on ``draws`` resamples of the held-out episodes with ``rng(0)`` (the same
    resamples for every encoder scored on the same windows); memoised per object."""
    key = (id(pred), draws)
    if key not in _BOOTS or _BOOTS[key][0] is not pred:
        means = np.array([d["mean"] for d in pred.bootstrap(draws, np.random.default_rng(0))])
        _BOOTS[key] = (pred, means)
    return _BOOTS[key][1]


def _family_var(scores: np.ndarray) -> float:
    """Sample variance of a family's scores (0 for a single deterministic encoder)."""
    return float(scores.var(ddof=1)) if len(scores) > 1 else 0.0


def contrast(
    a: list[Predictions], b: list[Predictions], paired: bool, draws: int = DRAWS
) -> Contrast:
    """``a - b`` from out-of-fold predictions on the same held-out windows.

    Args:
        a, b: One prediction set per encoder. With ``paired``, ``a[i]`` and ``b[i]`` share a seed
            (same initialisation or data order) and the seed SE is that of the paired
            differences; otherwise the two families are independent.
        paired: See above.
        draws: Bootstrap resamples of the held-out episodes.

    Raises:
        ValueError: if a family is empty, paired families differ in size, or any two encoders
            were not scored on the same windows.
    """
    if not a or not b or (paired and len(a) != len(b)):
        raise ValueError("contrast needs two non-empty families (of equal size if paired)")
    ref = a[0]
    for p in (*a, *b):
        if not np.array_equal(p.groups, ref.groups) or not np.array_equal(p.target, ref.target):
            raise ValueError("every encoder must be scored on the same held-out windows")
    sa = np.array([p.scores()["mean"] for p in a])
    sb = np.array([p.scores()["mean"] for p in b])
    if paired:
        diffs = sa - sb
        seed_se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    else:
        seed_se = float(np.sqrt(_family_var(sa) / len(sa) + _family_var(sb) / len(sb)))
    boot = np.mean([boot_means(p, draws) for p in a], 0) - np.mean(
        [boot_means(p, draws) for p in b], 0
    )
    return Contrast(float(sa.mean() - sb.mean()), seed_se, float(boot.std()), len(a), len(b))


def difference_in_differences(
    a: list[Predictions],
    b: list[Predictions],
    c: list[Predictions],
    d: list[Predictions],
    draws: int = DRAWS,
) -> Contrast:
    """``(a - b) - (c - d)`` of the probe mean, every family paired by seed (``a[i]``, ``b[i]``,
    ``c[i]`` and ``d[i]`` share seed ``i``) and all scored on the same held-out windows: e.g. the
    change of a data contrast when the sampler changes. Seed SE from the per-seed values, episode
    SE from the shared bootstrap.

    Raises:
        ValueError: if the families are empty or differ in size, or any two encoders were not
            scored on the same windows.
    """
    families = (a, b, c, d)
    if not a or len({len(f) for f in families}) != 1:
        raise ValueError("difference_in_differences needs four non-empty families of equal size")
    ref = a[0]
    for p in (x for f in families for x in f):
        if not np.array_equal(p.groups, ref.groups) or not np.array_equal(p.target, ref.target):
            raise ValueError("every encoder must be scored on the same held-out windows")
    m = [np.array([p.scores()["mean"] for p in f]) for f in families]
    per_seed = (m[0] - m[1]) - (m[2] - m[3])
    seed_se = float(per_seed.std(ddof=1) / np.sqrt(len(per_seed))) if len(per_seed) > 1 else 0.0
    boot = [np.mean([boot_means(p, draws) for p in f], 0) for f in families]
    episode = (boot[0] - boot[1]) - (boot[2] - boot[3])
    return Contrast(float(per_seed.mean()), seed_se, float(episode.std()), len(a), len(c))


def contrast_values(
    a: np.ndarray, b: np.ndarray, groups: np.ndarray, paired: bool = True, draws: int = DRAWS
) -> Contrast:
    """``a - b`` of per-item values (e.g. a success indicator or an error per goal), averaged
    over items, with the same seed and episode uncertainty as ``contrast``.

    Args:
        a, b: ``[seeds, n]`` values of the two families on the same ``n`` items (with
            ``paired``, row ``i`` of both shares a seed; otherwise the rows are independent and
            the counts may differ).
        groups: ``[n]`` episode of each item: the bootstrap resamples episodes with ``rng(0)``.
        paired: See above.
        draws: Bootstrap resamples.

    Raises:
        ValueError: on empty families, mismatched item counts, or unequal seed counts if paired.
    """
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if a.size == 0 or b.size == 0 or a.shape[1] != b.shape[1] or a.shape[1] != len(groups):
        raise ValueError("contrast_values needs two non-empty families on the same items")
    if paired and a.shape[0] != b.shape[0]:
        raise ValueError("paired families need the same number of seeds")
    ma, mb = a.mean(1), b.mean(1)
    if paired:
        d = ma - mb
        seed_se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    else:
        seed_se = float(np.sqrt(_family_var(ma) / len(ma) + _family_var(mb) / len(mb)))
    episodes = np.unique(groups)
    rows = {e: np.flatnonzero(groups == e) for e in episodes}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(draws):
        idx = np.concatenate([rows[e] for e in rng.choice(episodes, len(episodes))])
        boot.append(a[:, idx].mean() - b[:, idx].mean())
    return Contrast(float(ma.mean() - mb.mean()), seed_se, float(np.std(boot)), len(a), len(b))


def describe(c: Contrast) -> str:
    """``+0.012 (seed SE 0.010, episode SE 0.015, combined 0.018, MDD 0.050, resolved)``."""
    tag = ", resolved" if c.resolved else ""
    return (
        f"{c.point:+.3f} (seed SE {c.seed_se:.3f}, episode SE {c.episode_se:.3f}, "
        f"combined {c.se:.3f}, MDD {c.mdd:.3f}{tag})"
    )
