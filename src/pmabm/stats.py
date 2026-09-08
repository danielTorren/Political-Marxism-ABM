"""Cross-seed statistics: paired comparisons between arms, and outcome regimes.

Two ideas, both aimed at the same failure mode -- reporting a mean where a mean is the wrong
summary.

**Paired differences.** Every arm of a scenario group runs against the *same* seed list, which
the paper's Results preamble states as a convention: "an ablation is always reported against the
same seeds as its baseline, so that a difference between arms is not a difference between draws."
An independent-sample confidence interval on each arm separately throws that away. Seed *s*
differs between two arms only in the mechanism under test, so the difference *within* seed *s*
removes every source of variation the two arms share -- the estate layout, the initial customary
rents, the ecological shock sequence -- and what is left is the mechanism. In this model the
shared variation is large, so the paired interval is typically several times tighter than the
unpaired one on the same runs.

**Regimes.** A mean final Leasehold share of 0.5 can mean every run half-converted, or half the
runs converting fully and half not at all. Those are different claims about the model and the
paper's own preamble flags the distinction ("a tight interval on a bimodal distribution is a
statement about arithmetic rather than about the model"). :func:`regime_shares` reports the
fraction of seeds in each outcome class instead, which is the summary that survives bimodality --
and it is the right object for RQ4, whose question is whether England still *arrives*, not what
its average share is.

No scipy: the sign test is exact from :func:`math.comb`, which is well within range at the seed
counts used here.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

#: Confidence multiplier, matching the 1.96 used throughout the plotting modules rather than a
#: per-n Student's t. At 32 seeds the exact two-sided 95% multiplier is 2.04, so intervals here
#: are about 4% narrower than they should strictly be; the alternative was one convention in this
#: module and a different one in every figure it feeds, which is the worse error.
Z95 = 1.96

#: Boundaries of the outcome classes, on the final Leasehold share of tenancies. The upper one is
#: the paper's own completion criterion -- a Leasehold majority, drawn as the dashed line in the
#: RQ4 figure -- so "transitioned" here means what the paper means by the transition completing.
REGIME_EDGES = (0.2, 0.5)
REGIME_LABELS = ("failed", "partial", "transitioned")


def sign_test(deltas: np.ndarray) -> float:
    """Two-sided exact sign test that the median difference is zero.

    Reported alongside the paired interval because it assumes nothing about the shape of the
    differences, which matters here: an ablation that flips some seeds from full transition to
    none and leaves the rest alone produces a wildly non-normal difference distribution, and the
    interval on its mean is then hard to read while the sign test is not.

    Ties are dropped, which is the conventional treatment and the conservative one -- a run where
    the ablation changed nothing is evidence for the null and is simply not counted either way.
    """
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d) & (d != 0.0)]
    n = len(d)
    if n == 0:
        return float("nan")
    k = int((d > 0).sum())
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


def paired_delta(
    a: pd.DataFrame,
    b: pd.DataFrame,
    columns: list[str],
    label_a: str = "a",
    label_b: str = "b",
    seed_column: str = "seed",
) -> pd.DataFrame:
    """Per-seed differences ``a - b`` on the seeds both arms ran, one row per column.

    Both frames are one row per seed -- the ``summary`` frame of a scenario arm. Seeds present in
    only one arm are dropped rather than filled: a partial arm should narrow the comparison, not
    silently contribute an unpaired observation to it.

    Returns ``mean_delta`` with a paired 95% interval, the unpaired interval beside it so the gain
    from pairing is visible, the sign-test p-value, and both arm means.
    """
    left = a.set_index(seed_column)
    right = b.set_index(seed_column)
    shared = left.index.intersection(right.index)
    rows = []
    for column in columns:
        if column not in left.columns or column not in right.columns:
            continue
        x = left.loc[shared, column].astype(float)
        y = right.loc[shared, column].astype(float)
        delta = (x - y).dropna()
        n = len(delta)
        if n == 0:
            continue
        sd = float(delta.std(ddof=1)) if n > 1 else 0.0
        ci = Z95 * sd / np.sqrt(n) if n > 1 else 0.0
        # The interval the same runs would have produced treated as two independent samples,
        # carried so a figure can show what pairing bought rather than asserting it.
        xs, ys = x.dropna(), y.dropna()
        unpaired = (
            Z95 * np.sqrt(xs.var(ddof=1) / len(xs) + ys.var(ddof=1) / len(ys))
            if len(xs) > 1 and len(ys) > 1
            else np.nan
        )
        rows.append(
            {
                "metric": column,
                "arm_a": label_a,
                "arm_b": label_b,
                "n_pairs": n,
                "mean_a": float(x.mean()),
                "mean_b": float(y.mean()),
                "mean_delta": float(delta.mean()),
                "sd_delta": sd,
                "ci_paired": float(ci),
                "ci_unpaired": float(unpaired) if np.isfinite(unpaired) else np.nan,
                "lo": float(delta.mean() - ci),
                "hi": float(delta.mean() + ci),
                "p_sign": sign_test(delta.to_numpy()),
                # The whole point of pairing, as a number: >1 means the paired interval is
                # tighter. Values of 3-5 are normal in this model, where seeds differ a great
                # deal and arms within a seed differ much less.
                "pairing_gain": (
                    float(unpaired / ci) if np.isfinite(unpaired) and ci > 0 else np.nan
                ),
                "significant": bool(np.isfinite(ci) and abs(delta.mean()) > ci),
            }
        )
    return pd.DataFrame(rows)


def paired_delta_table(
    arms: dict[str, pd.DataFrame], baseline: str, columns: list[str]
) -> pd.DataFrame:
    """:func:`paired_delta` of every arm against one named baseline, stacked.

    ``arms`` maps a display label to that arm's ``summary`` frame. The baseline is compared
    against itself too, which gives a row of exact zeros -- kept deliberately, because it is the
    visual anchor of the forest plot and its absence would leave the reader to supply the zero
    line from the axis alone.
    """
    if baseline not in arms:
        return pd.DataFrame()
    parts = [
        paired_delta(frame, arms[baseline], columns, label_a=label, label_b=baseline)
        for label, frame in arms.items()
    ]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def classify_regimes(
    summary: pd.DataFrame,
    column: str = "final_share_leasehold",
    edges: tuple[float, float] = REGIME_EDGES,
) -> pd.Series:
    """Label each seed ``failed`` / ``partial`` / ``transitioned`` on its final outcome."""
    values = summary[column].astype(float)
    return pd.Series(
        pd.cut(
            values,
            bins=[-np.inf, edges[0], edges[1], np.inf],
            labels=list(REGIME_LABELS),
            right=False,
        ),
        index=summary.index,
        name="regime",
    )


def regime_shares(
    arms: dict[str, pd.DataFrame],
    column: str = "final_share_leasehold",
    edges: tuple[float, float] = REGIME_EDGES,
) -> pd.DataFrame:
    """Fraction of seeds in each outcome class per arm, with the mean beside it.

    The mean is carried alongside on purpose: the pair is what shows a mean to be misleading. An
    arm at mean 0.5 with every seed ``partial`` and one at mean 0.5 split evenly between
    ``failed`` and ``transitioned`` are different models, and only this table separates them.
    """
    rows = []
    for label, frame in arms.items():
        if column not in frame.columns:
            continue
        regimes = classify_regimes(frame, column, edges)
        counts = regimes.value_counts()
        n = int(counts.sum())
        if n == 0:
            continue
        row = {"arm": label, "n_seeds": n, "mean": float(frame[column].mean())}
        for name in REGIME_LABELS:
            row[f"share_{name}"] = float(counts.get(name, 0) / n)
            row[f"n_{name}"] = int(counts.get(name, 0))
        # A crude but sufficient bimodality flag: mass at both extremes and little in between is
        # exactly the case where the mean should not be quoted without this table beside it.
        row["bimodal"] = bool(
            row["share_failed"] > 0.2
            and row["share_transitioned"] > 0.2
            and row["share_partial"] < 0.4
        )
        rows.append(row)
    return pd.DataFrame(rows)


def transition_probability(
    summary: pd.DataFrame,
    column: str = "final_share_leasehold",
    threshold: float = REGIME_EDGES[1],
) -> tuple[float, float]:
    """``(P(transition), 95% interval half-width)`` across seeds.

    A binomial proportion with a normal interval, which is the right object for RQ4: the question
    is whether England still arrives at a given level of customary security, not what its average
    Leasehold share is. Reported as a proportion of *seeds*, so the interval narrows with
    replication rather than with lattice size.
    """
    values = summary[column].astype(float).dropna()
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    p = float((values >= threshold).mean())
    return p, float(Z95 * np.sqrt(max(p * (1.0 - p), 0.0) / n))
