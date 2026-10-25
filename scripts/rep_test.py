"""B-rep-0: can the state s carry the arm? (reconstruction on s, several summary tokens)

Pre-registration and outcome rules: ``docs/model.md``, "B-rep-0". Four arms on B-fix-1's recipe
(fix1: one masked pass, reconstruction weight 0.5), 10,000 steps, seeds 0 to 2:

* ``recon-s``: the reconstruction decoder reads the state s instead of the summary z;
* ``recon-s-allprop`` (1b): as ``recon-s``, and the proprio reconstruction is scored on every
  channel against the clean input (two changes against fix1);
* ``multi-z``: four summary tokens, s their projected concatenation (512-d);
* ``recon-s-multi-z``: both.

Baselines, not retrained: B-variety-0's ``jepa-babble`` (A0, from ``runs/variety``) and B-fix-1's
``fix1-recon05``, whose checkpoints, logs and probe predictions are copied into ``runs/rep`` (the
probe battery reuses them: encoders already probed are skipped).

Measures: B-plan-0's decoder check (``plan_test.py``: ridge arm decoders on z, s and each
projected token; train, CV and held-out R2), its open loop with a per-effector decoder and copy,
and B-fix-0/1's probe battery (``fix_test.py``) at every checkpoint.

    uv run python scripts/rep_test.py train --arm recon-s --seed 0
    uv run python scripts/rep_test.py encode --shard 0/16    # probe battery
    uv run python scripts/rep_test.py probe --shard 0/16
    uv run python scripts/rep_test.py fit                    # ridge decoders (pooled, per effector)
    uv run python scripts/rep_test.py decoder --shard 0/16   # decoder check
    uv run python scripts/rep_test.py openloop --shard 0/5   # own-predictor open loop
    uv run python scripts/rep_test.py report
    uv run python scripts/rep_test.py locate --shard 0/9     # post hoc: where is the arm lost?
    uv run python scripts/rep_test.py locate-report
    uv run python scripts/rep_test.py tokens-dyn --shard 0/6 # design step (a): fused-token dynamics
    uv run python scripts/rep_test.py tokens-base            # model-free baselines with actions
    uv run python scripts/rep_test.py tokens-dyn-report
"""

import argparse
import json
import logging
from pathlib import Path

import fix_test as fix  # scripts/ is on sys.path when this file runs as a script
import matplotlib
import numpy as np
import plan_test as plan
import torch
import variety_test as variety

from sensorium.model.paired import Contrast, describe

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("runs/rep")
FIX1 = "fix1-recon05"
FIX1_OPTIONS: dict[str, object] = {"recon_weight": 0.5, "recon_input": "masked"}
ARMS: dict[str, dict[str, object]] = {
    "recon-s": FIX1_OPTIONS | {"recon_source": "s"},
    "recon-s-allprop": FIX1_OPTIONS | {"recon_source": "s", "recon_proprio": "all"},
    "multi-z": FIX1_OPTIONS | {"summary_tokens": 4},
    "recon-s-multi-z": FIX1_OPTIONS | {"recon_source": "s", "summary_tokens": 4},
}
"""The arms' model options (``ModelConfig``) on fix1's."""
LOCATE_ARMS = ("A0", FIX1, "multi-z")
"""The encoders of the post hoc locate check (where the arm is lost)."""
TOKENS_ARMS = ("A0", FIX1)
"""The encoders of design step (a), linear dynamics on the fused tokens; a later study adds its
checkpoints to ``plan.ENCODERS`` and here."""
FIXED_R2 = 0.85
"""The "fixed" bar: mean per-effector R2 of s, on train and on held-out babble_v2."""
STEP = variety.STEPS


def configure() -> None:
    """Point B-fix's training and probe code and B-plan-0's decoder and open-loop code at this
    study's runs (``runs/rep``): the new arms, the fix1 baseline and A0."""
    fix.ROOT = ROOT
    fix.ARMS = {FIX1: FIX1_OPTIONS, **ARMS}
    plan.ROOT = ROOT / "plan"
    plan.ENCODERS = {
        "A0": (variety.ROOT / "runs", "jepa-babble"),
        **{arm: (ROOT / "runs", arm) for arm in (FIX1, *ARMS)},
    }
    plan.OWN = ("A0", FIX1, *ARMS)
    plan.RIDGE = ()
    plan.LOCATE_ARMS = LOCATE_ARMS
    plan.TOKENS_ARMS = TOKENS_ARMS
    plan.layer.cache_clear()
    plan.heads.cache_clear()
    plan.effector_decoders.cache_clear()


# ---------------------------------------------------------------------------------------------
# report


def reading(c: Contrast) -> str:
    """The pre-registered reading of ``arm - fix1``."""
    if not c.resolved:
        return "no difference"
    return "improved" if c.point > 0 else "worse"


def decoder_runs(arm: str) -> list[tuple[dict, dict[str, np.ndarray]]]:
    """An encoder's decoder-check results per seed: the R2 json and the held-out predictions."""
    out = []
    for seed in plan.SEEDS:
        path = plan.ROOT / "decoder" / f"{arm}-s{seed}.json"
        if path.exists():
            out.append((json.loads(path.read_text()), dict(np.load(path.with_suffix(".npz")))))
    return out


def best_token(runs: list[tuple[dict, dict[str, np.ndarray]]]) -> str | None:
    """The projected token with the highest mean per-effector held-out babble_v2 R2."""
    tokens = plan.token_features(runs[0][0]) if runs else []
    if not tokens:
        return None
    return max(tokens, key=lambda t: np.mean([r[t]["per effector"]["babble"] for r, _ in runs]))


def r2_contrast(arm: str, feature: str, test: str = "babble") -> Contrast | None:
    """``R2(arm, feature) - R2(fix1, s)``, per-effector decoders on a held-out set, paired by
    seed with an episode bootstrap (``None`` unless both have every seed)."""
    a, b = decoder_runs(arm), decoder_runs(FIX1)
    if len(a) != len(plan.SEEDS) or len(b) != len(plan.SEEDS):
        return None
    q, groups = b[0][1][f"q/{test}"], b[0][1][f"groups/{test}"]
    return plan.r2_contrast_values(
        [p[f"{feature}/per effector/{test}"] for _, p in a],
        [p[f"s/per effector/{test}"] for _, p in b],
        q,
        groups,
    )


def report_primary() -> None:
    """The primary contrasts, the "fixed" bar and the best-single-token control."""
    variety.say("\n## Primary: per-effector R2 of s, held-out babble_v2, arm - fix1\n")
    variety.say("| arm | R2 of s | contrast | reading | fixed (>= 0.85 train and held-out) |")
    variety.say("|---|---|---|---|---|")
    for arm in (FIX1, *ARMS):
        runs = decoder_runs(arm)
        if not runs:
            continue
        held = np.array([r["s"]["per effector"]["babble"] for r, _ in runs])
        train = np.array([r["s"]["per effector"]["train"] for r, _ in runs])
        fixed = "yes" if min(held.mean(), train.mean()) >= FIXED_R2 else "no"
        c = r2_contrast(arm, "s") if arm != FIX1 else None
        cells = (describe(c), reading(c)) if c is not None else ("-", "-")
        variety.say(
            f"| {arm} | {held.mean():.3f} +- {held.std():.3f} | {cells[0]} | {cells[1]} | {fixed} |"
        )
    variety.say("\nCapacity against structure: the best single projected token (128-d)\n")
    variety.say("| arm | token | R2 of the token | token - fix1 s |\n|---|---|---|---|")
    for arm in ARMS:
        runs = decoder_runs(arm)
        token = best_token(runs)
        if token is None:
            continue
        r2s = np.array([r[token]["per effector"]["babble"] for r, _ in runs])
        c = r2_contrast(arm, token)
        variety.say(
            f"| {arm} | {token} | {r2s.mean():.3f} +- {r2s.std():.3f} | "
            + (f"{describe(c)} ({reading(c)})" if c is not None else "-")
            + " |"
        )


def report_openloop() -> None:
    """Open loop on held-out babble_v2 with the per-effector decoder: the decoded true state
    (ceiling), the own-predictor rollout and copy per horizon; "useful for planning" if the
    horizon-8 rollout is below copy."""
    copy = plan.copy_arm()[0]
    variety.say("\n## Open loop, held-out babble_v2, per-effector decoder (rad)\n")
    cols = [h - 1 for h in plan.HORIZONS]
    head = " | ".join(f"h{h}" for h in plan.HORIZONS)
    variety.say(f"| arm | line | {head} | useful (h8 < copy) |")
    variety.say("|---|---|" + "---|" * (len(cols) + 1))
    curves = {}
    babble = None
    for arm in plan.OWN:
        runs = [r for r in plan.load_openloop("own", arm) if "decoded_eff" in r]
        if not runs:
            continue
        babble = runs[0]["set"] == 0
        pred = np.array([r["decoded_eff"][babble].mean(0) for r in runs])
        ceil = np.array([r["decoded_true_eff"][babble].mean(0) for r in runs])
        curves[arm] = pred
        useful = "yes" if pred[:, -1].mean() < copy[babble, -1].mean() else "no"
        for line, v, flag in (("ceiling", ceil, ""), ("own prediction", pred, useful)):
            cells = " | ".join(f"{v[:, c].mean():.3f} +- {v[:, c].std():.3f}" for c in cols)
            variety.say(f"| {arm} | {line} | {cells} | {flag} |")
    if babble is None:
        return
    base = copy[babble].mean(0)
    variety.say("| copy | - | " + " | ".join(f"{base[c]:.3f}" for c in cols) + " | |")
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm, pred in curves.items():
        ax.errorbar(
            plan.HORIZONS,
            pred[:, cols].mean(0),
            pred[:, cols].std(0),
            marker="o",
            capsize=2,
            label=arm,
        )
    ax.plot(plan.HORIZONS, base[cols], color="0.4", ls=":", label="copy")
    ax.set(
        xscale="log",
        xlabel="horizon (model steps of 0.2 s)",
        ylabel="arm RMS (rad)",
        title="own prediction, per-effector decoder, held-out babble_v2",
    )
    ax.set_xticks(plan.HORIZONS, [str(h) for h in plan.HORIZONS])
    ax.minorticks_off()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "rep_openloop.png")


def r2_figure() -> None:
    """``rep_r2.png``: R2 of s and z, pooled and per effector, on train, held-out babble_v2 and
    held-out variety, per encoder (mean +- sd over seeds), with the raw proprio window."""
    variants = (("z", "pooled"), ("z", "per effector"), ("s", "pooled"), ("s", "per effector"))
    colors = ("#9ecae1", "#2171b5", "#fdae6b", "#d94801")
    arms = [a for a in plan.ENCODERS if decoder_runs(a)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    w = 0.2
    for ax, (key, title) in zip(
        axes,
        (("train", "train"), ("babble", "held-out babble_v2"), ("variety", "held-out variety")),
        strict=True,
    ):
        for i, ((feature, dec), color) in enumerate(zip(variants, colors, strict=True)):
            vals = np.array([[r[feature][dec][key] for r, _ in decoder_runs(a)] for a in arms])
            x = np.arange(len(arms)) + (i - 1.5) * w
            ax.bar(
                x,
                vals.mean(1),
                w * 0.92,
                yerr=vals.std(1),
                capsize=2,
                color=color,
                label=f"{feature}, {dec}",
                edgecolor="white",
                linewidth=0.5,
            )
        ax.axhline(FIXED_R2, color="black", lw=0.8, ls="--", label=f"fixed bar ({FIXED_R2})")
        ax.set_xticks(np.arange(len(arms)), arms, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("arm-joint R²")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "rep_r2.png")


def save(fig: plt.Figure, name: str) -> None:
    """Save under ``runs/rep/figures`` and close."""
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures" / name, dpi=130)
    plt.close(fig)


def report_battery() -> None:
    """Secondary: the probe battery of each arm against fix1 and A0 at 10,000 steps (paired by
    seed and episode), and the training summary."""
    variety.say("\n## Secondary: probe battery at 10,000 steps (arm - fix1, arm - A0)\n")
    variety.say("| arm | set | score | layer | arm - fix1 | arm - A0 |\n|---|---|---|---|---|---|")
    for arm in ARMS:
        for label in fix.EVAL_SETS:
            for score in fix.SCORES:
                for layer in fix.LAYERS:
                    a = fix.family(label, score, layer, arm, STEP)
                    cells = []
                    for ref in (FIX1, "A0"):
                        c = fix.paired(a, fix.family(label, score, layer, ref, STEP))
                        cells.append(describe(c) if c is not None else "-")
                    variety.say(
                        f"| {arm} | {label} | {score} | {layer} | " + " | ".join(cells) + " |"
                    )
    variety.say("\nTraining (reported, not judged): seed means of the last 500 steps\n")
    variety.say("| arm | next-state MSE | recon | seq/s |\n|---|---|---|---|")
    for arm in (FIX1, *ARMS):
        t = fix.training_summary(arm)
        if t:
            variety.say(f"| {arm} | {t['pred']:.4f} | {t['recon']:.4f} | {t['seq_per_s']:.1f} |")


def report() -> None:
    """Every table and figure of B-rep-0."""
    variety.say("B-rep-0 (pre-registered in docs/model.md).")
    plan.decoder_report()
    report_primary()
    r2_figure()
    report_openloop()
    report_battery()


SHARDED = {
    "encode": lambda shard: fix.encode(shard),
    "probe": lambda shard: fix.probe(shard),
    "decoder": lambda shard: plan.decoder(shard),
    "openloop": lambda shard: plan.openloop(shard),
    "locate": lambda shard: plan.locate(shard),
    "tokens-dyn": lambda shard: plan.tokens_dyn(shard),
}
"""The commands that take ``--shard`` (looked up at call time, after ``configure``)."""
WHOLE = {
    "fit": lambda: plan.fit(),
    "report": lambda: report(),
    "locate-report": lambda: plan.locate_report(),
    "tokens-dyn-report": lambda: plan.tokens_report(),
    "tokens-base": lambda: plan.tokens_base(),
}
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
        fix.train_arm(args.arm, args.seed, args.steps, args.workers)
    elif args.command in SHARDED:
        SHARDED[args.command](args.shard)
    else:
        WHOLE[args.command]()


if __name__ == "__main__":
    main()
