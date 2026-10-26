"""B-skip-0: do skip connections let the state s keep the arm? (a representation test)

Pre-registration: the code PR's body (and ``docs/model.md`` once written up). Three arms on fix1's
recipe (B-rep-0's data and schedule), 10,000 steps, seeds 0 to 2:

* ``skip-z``: ``z = summary + W [fused sense tokens] + b`` (``model.skip_tokens``; W, b start at 0);
* ``skip-proj``: ``s = projector(z) + z`` (``model.skip_projector``);
* ``skip-both``: both.

Controls, not retrained: fix1 (B-fix-1 ``fix1-recon05``) and A0 (B-variety-0 ``jepa-babble``). This
script reuses B-rep-0's machinery (``rep_test.py``), pointed at ``runs/skip``:

* primary: the decoder check (per-effector R2 of s on held-out babble_v2), arm - fix1;
* secondary: the locate stages (with every fused sense token, the ceiling of the gap), the ridge
  open loop on s against the arm and action copies (tokens-dyn, tokens-base), the probe battery,
  the training curves and the sense reliance of the state (does s keep vision?).

    uv run python scripts/skip_test.py train --arm skip-z --seed 0
    uv run python scripts/skip_test.py encode --shard 0/12     # probe battery
    uv run python scripts/skip_test.py probe --shard 0/12
    uv run python scripts/skip_test.py decoder --shard 0/16    # primary
    uv run python scripts/skip_test.py locate --shard 0/15
    uv run python scripts/skip_test.py tokens-dyn --shard 0/12
    uv run python scripts/skip_test.py tokens-base
    uv run python scripts/skip_test.py reliance --shard 0/15
    uv run python scripts/skip_test.py report
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import rep_test as rep  # scripts/ is on sys.path when this file runs as a script
import torch

from sensorium.model.diagnostics import sense_reliance
from sensorium.model.paired import describe

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("runs/skip")
FIX1 = rep.FIX1
ARMS: dict[str, dict[str, object]] = {
    "skip-z": rep.FIX1_OPTIONS | {"skip_tokens": True},
    "skip-proj": rep.FIX1_OPTIONS | {"skip_projector": True},
    "skip-both": rep.FIX1_OPTIONS | {"skip_tokens": True, "skip_projector": True},
}
"""The arms' model options on fix1's."""
RELIANCE_EPISODES = 40
"""Held-out babble_v2 episodes for the sense reliance (stride 16)."""
CURVES = ("pred", "reg", "recon", "erank")


def configure() -> None:
    """Point B-rep-0's code (and through it B-fix's and B-plan-0's) at this study's runs."""
    rep.ROOT = ROOT
    rep.ARMS = ARMS
    rep.LOCATE_ARMS = ("A0", FIX1, *ARMS)
    rep.TOKENS_ARMS = (FIX1, *ARMS)
    rep.configure()


def encoders() -> tuple[str, ...]:
    """Every encoder of the study: the controls, then the arms."""
    return ("A0", FIX1, *ARMS)


# ---------------------------------------------------------------------------------------------
# sense reliance


def reliance(shard: str) -> None:
    """How far removing each sense moves the state (``diagnostics.sense_reliance``), per
    encoder and seed, on the first ``RELIANCE_EPISODES`` held-out babble_v2 episodes."""
    out = ROOT / "reliance"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(a, s) for a in encoders() for s in rep.plan.SEEDS]
    i, n = (int(x) for x in shard.split("/"))
    for arm, seed in jobs[i::n]:
        path = out / f"{arm}-s{seed}.json"
        if path.exists():
            continue
        logging.info("reliance: %s-s%d started", arm, seed)
        paths = rep.variety.files("babble", "heldout")[:RELIANCE_EPISODES]
        path.write_text(json.dumps(sense_reliance(rep.plan.layer(arm, seed), paths, stride=16)))
        logging.info("reliance: %s-s%d done, peak RSS %.0f MB", arm, seed, rep.plan.peak_rss_mb())


def report_reliance() -> None:
    """The sense reliance per encoder (mean +- sd over seeds)."""
    rows = {}
    for arm in encoders():
        paths = [ROOT / "reliance" / f"{arm}-s{s}.json" for s in rep.plan.SEEDS]
        runs = [json.loads(p.read_text()) for p in paths if p.exists()]
        if runs:
            rows[arm] = runs
    if not rows:
        return
    senses = sorted(next(iter(rows.values()))[0])
    variety = rep.variety
    variety.say("\n## Secondary: sense reliance of s (held-out babble_v2; 0 = ignores the sense)\n")
    variety.say("| encoder | " + " | ".join(s.removeprefix("reliance_") for s in senses) + " |")
    variety.say("|---|" + "---|" * len(senses))
    for arm, runs in rows.items():
        cells = [rep.plan.mean_sd([r[s] for r in runs]) for s in senses]
        variety.say(f"| {arm} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------------------------
# report


def decoder_r2(arm: str, feature: str = "s") -> np.ndarray:
    """Per-seed per-effector held-out babble_v2 R2 of a feature (decoder check)."""
    return np.array([r[feature]["per effector"]["babble"] for r, _ in rep.decoder_runs(arm)])


def fused_ceiling() -> float | None:
    """fix1's per-effector held-out babble_v2 R2 of every fused sense token (locate check), the
    ceiling of the gap; ``None`` if not computed."""
    paths = [rep.plan.ROOT / "locate" / f"{FIX1}-s{s}.json" for s in rep.plan.SEEDS]
    runs = [json.loads(p.read_text()) for p in paths if p.exists()]
    if not runs:
        return None
    return float(np.mean([r["fused_all"]["per effector"]["babble"] for r in runs]))


def report_primary() -> None:
    """The pre-registered primary: arm - fix1 on the per-effector R2 of s (held-out
    babble_v2), paired by seed with an episode bootstrap; "keeps the arm" if the gain is above
    the MDD and positive in every seed; and the share of the gap to fix1's fused-token ceiling
    that each arm closes."""
    variety = rep.variety
    base = decoder_r2(FIX1)
    ceiling = fused_ceiling()
    variety.say("\n## Primary: per-effector R2 of s, held-out babble_v2, arm - fix1\n")
    if ceiling is not None:
        variety.say(f"fix1 s {base.mean():.3f}; fix1 fused sense tokens (ceiling) {ceiling:.3f}\n")
    variety.say("| arm | R2 of s | arm - fix1 | every seed | keeps the arm | gap closed |")
    variety.say("|---|---|---|---|---|---|")
    for arm in ARMS:
        r2s = decoder_r2(arm)
        c = rep.r2_contrast(arm, "s")
        if c is None or len(r2s) != len(base):
            continue
        every = bool(np.all(r2s - base > 0))
        keeps = "yes" if c.point > c.mdd and every else "no"
        gap = f"{(r2s.mean() - base.mean()) / (ceiling - base.mean()):.2f}" if ceiling else "-"
        variety.say(
            f"| {arm} | {rep.plan.mean_sd(list(r2s))} | {describe(c)} | "
            f"{'yes' if every else 'no'} | {keeps} | {gap} |"
        )


def loss_figure() -> None:
    """``figures/skip_loss.png``: the training curves of every seed (one line each) of the arms,
    fix1 and A0 (5-point moving average)."""
    runs = rep.fix.collect_logs()
    fig, axes = plt.subplots(1, len(CURVES), figsize=(4.5 * len(CURVES), 3.6))
    for i, arm in enumerate(encoders()):
        for path in sorted(runs.glob(f"{arm}-s*/log.jsonl")):
            log = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            steps = np.array([r["step"] for r in log])
            for ax, key in zip(axes, CURVES, strict=True):
                v = np.array([r.get(key, np.nan) for r in log], dtype=float)
                if np.isnan(v).all():
                    continue
                w = min(5, len(v))
                smooth = np.convolve(v, np.ones(w) / w, mode="valid")
                label = arm if path.parent.name.endswith("-s0") else None
                ax.plot(steps[w - 1 :], smooth, color=f"C{i}", lw=0.8, label=label)
    for ax, key in zip(axes, CURVES, strict=True):
        ax.set(title=key, xlabel="step")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    rep.save(fig, "skip_loss.png")


def report() -> None:
    """Every table and figure of B-skip-0."""
    rep.variety.say("B-skip-0 (pre-registered in the code PR).")
    rep.plan.decoder_report()
    report_primary()
    rep.r2_figure()
    rep.plan.locate_report()
    rep.plan.tokens_report(judged=False)
    rep.report_battery()
    report_reliance()
    loss_figure()


SHARDED = {
    "encode": lambda shard: rep.fix.encode(shard),
    "probe": lambda shard: rep.fix.probe(shard),
    "decoder": lambda shard: rep.plan.decoder(shard),
    "locate": lambda shard: rep.plan.locate(shard),
    "tokens-dyn": lambda shard: rep.plan.tokens_dyn(shard),
    "reliance": lambda shard: reliance(shard),
}
"""The commands that take ``--shard`` (looked up at call time, after ``configure``)."""
WHOLE = {"tokens-base": lambda: rep.plan.tokens_base(), "report": lambda: report()}
"""The commands that run whole."""


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=("train", *SHARDED, *WHOLE))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    configure()
    if args.command == "train":
        if args.arm is None:
            parser.error("train needs --arm")
        rep.fix.train_arm(args.arm, args.seed, args.steps, args.workers)
    elif args.command in SHARDED:
        SHARDED[args.command](args.shard)
    else:
        WHOLE[args.command]()


if __name__ == "__main__":
    main()
