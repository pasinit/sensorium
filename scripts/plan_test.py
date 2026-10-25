"""B-plan-0: does the reconstruction default cost the JEPA its use as a world model?

Pre-registration and outcome rules: ``docs/model.md``, "B-plan-0". Encoders: the 10,000-step
checkpoints (seeds 0 to 2) of A0 (B-variety-0 ``jepa-babble``), recon05 (B-fix-0), fix1
(B-fix-1 ``fix1-recon05``) and MAE (B-variety-0 ``mae-babble``), with their own predictors (not
MAE) and with a ridge dynamics head fitted on 100 babble_v2 pretraining episodes.

    uv run python scripts/plan_test.py fit                  # ridge heads, decoders, action stats
    uv run python scripts/plan_test.py fidelity             # replay gate on 20 goals (stage 1)
    uv run python scripts/plan_test.py openloop --shard 0/8 # (a) open-loop prediction
    uv run python scripts/plan_test.py goals --shard 0/30   # (b) closed-loop goal reaching
    uv run python scripts/plan_test.py rank --shard 0/30    # (b) proxy, if the gate fails
    uv run python scripts/plan_test.py report
    uv run python scripts/plan_test.py decoder --shard 0/13 # post hoc: arm decoder R2 (z, s, raw)
    uv run python scripts/plan_test.py decoder-report

Every stage skips work already saved, so a preempted VM resumes where it stopped.
"""

import argparse
import json
import logging
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path

import matplotlib
import numpy as np
import torch
import variety_test as variety  # scripts/ is on sys.path when this file runs as a script

from sensorium.bodies.restore import RestoredScene, block_settings
from sensorium.episode import Episode
from sensorium.model.data import load_episode, spacing
from sensorium.model.layout import Header
from sensorium.model.paired import Contrast, contrast_values, describe
from sensorium.model.planning import (
    ARM_JOINTS,
    Ridge,
    RidgeDynamics,
    arm_q,
    cem,
    chunks_at,
    encode_at,
    own_rollout,
    rms,
    sliding_ends,
    states_at,
    union_actions,
)
from sensorium.model.pretrain import load_layer
from sensorium.model.probes import episode_folds
from sensorium.model.sensory_layer import SensoryLayer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("runs/plan")
ENCODERS = {
    "A0": (Path("runs/variety/runs"), "jepa-babble"),
    "recon05": (Path("runs/fix/runs"), "recon05"),
    "fix1": (Path("runs/fix1/runs"), "fix1-recon05"),
    "MAE": (Path("runs/variety/runs"), "mae-babble"),
}
OWN = ("A0", "recon05", "fix1")
RIDGE = ("A0", "fix1", "MAE")
BARS = ("random", "hold", "replay")
STEP = 10_000
SEEDS = (0, 1, 2)
HORIZON = 8
HORIZONS = (1, 2, 4, 8)
STARTS = 8
GOALS_PER_EPISODE = 2
SUCCESS_RAD = 0.10
CURVE = np.round(np.linspace(0.05, 0.30, 11), 3)
FIDELITY_GOALS = 20
FIDELITY_BAR = 0.90
CANDIDATES = 64
TEST_SETS = ("babble", "variety")
CONFIGS = {"babble": Path("configs/babble_v2.toml"), "variety": Path("configs/variety.toml")}


# ---------------------------------------------------------------------------------------------
# encoders and data


@cache
def layer(arm: str, seed: int) -> SensoryLayer:
    """An arm's 10,000-step encoder."""
    root, run = ENCODERS[arm]
    return load_layer(root / f"{run}-s{seed}" / f"ckpt-{STEP}.pt")


def fit_paths() -> list[Path]:
    """The ridge heads' and decoders' episodes: the first 100 babble_v2 pretraining episodes."""
    return variety.files("babble", "pretrain", 100)


def model_stride(episode: Episode) -> int:
    """Env steps per model step (``K`` observations, ``spacing`` apart)."""
    return 4 * spacing(episode.dt)


def first_end(episode: Episode) -> int:
    """The first step a window can end at."""
    return 3 * spacing(episode.dt)


def arm_action_cols(episode: Episode) -> np.ndarray:
    """Columns of ``ARM_JOINTS`` in the episode's action vector."""
    names = list(episode.action_names)
    return np.array([names.index(n.removeprefix("q_")) for n in ARM_JOINTS])


# ---------------------------------------------------------------------------------------------
# fit


def fit() -> None:
    """Ridge dynamics and arm decoders for every encoder; arm-action statistics per effector."""
    out = ROOT / "fit"
    out.mkdir(parents=True, exist_ok=True)
    episodes = [Episode.load(p, privileged=False) for p in fit_paths()]
    stats: dict[str, list[np.ndarray]] = defaultdict(list)
    for ep in episodes:
        stats[str(ep.meta.get("effector", "hand"))].append(
            np.asarray(ep.action)[:, arm_action_cols(ep)]
        )
    summary = {
        eff: {
            "std": np.concatenate(v).std(0).tolist(),
            "low": np.concatenate(v).min(0).tolist(),
            "high": np.concatenate(v).max(0).tolist(),
        }
        for eff, v in stats.items()
    }
    (out / "action_stats.json").write_text(json.dumps(summary, indent=1))
    for arm in ENCODERS:
        for seed in SEEDS:
            path = out / f"{arm}-s{seed}.npz"
            if path.exists():
                continue
            lay = layer(arm, seed)
            s, c, n, q, g = [], [], [], [], []
            for i, ep in enumerate(episodes):
                ends = np.arange(first_end(ep), ep.length - model_stride(ep), model_stride(ep))
                st = states_at(lay, ep, np.append(ends, ends[-1] + model_stride(ep)))
                s.append(st[:-1])
                n.append(st[1:])
                c.append(chunks_at(lay, ep, ends))
                q.append(arm_q(ep, np.append(ends, ends[-1] + model_stride(ep)))[:-1])
                g.append(np.full(len(ends), i))
            states, groups = np.concatenate(s), np.concatenate(g)
            dyn = RidgeDynamics.fit(states, np.concatenate(c), np.concatenate(n), groups)
            dec = Ridge.fit(states, np.concatenate(q), groups)
            np.savez(
                path,
                **{f"dyn_{k}": getattr(dyn.ridge, k) for k in ("mean", "scale", "w", "bias")},
                **{f"dec_{k}": getattr(dec, k) for k in ("mean", "scale", "w", "bias")},
            )
            logging.info("fitted the ridge head and decoder of %s-s%d", arm, seed)


@cache
def heads(arm: str, seed: int) -> tuple[RidgeDynamics, Ridge]:
    """The fitted ridge dynamics and arm decoder of an encoder."""
    raw = np.load(ROOT / "fit" / f"{arm}-s{seed}.npz")
    dyn = Ridge(*(raw[f"dyn_{k}"] for k in ("mean", "scale", "w", "bias")))
    dec = Ridge(*(raw[f"dec_{k}"] for k in ("mean", "scale", "w", "bias")))
    return RidgeDynamics(dyn), dec


def rollout(
    predictor: str, arm: str, seed: int, start: np.ndarray, chunks: np.ndarray
) -> np.ndarray:
    """``[B, H, D]`` predicted states with the arm's own predictor or its ridge head."""
    if predictor == "own":
        return own_rollout(layer(arm, seed), start, chunks)
    return heads(arm, seed)[0].rollout(start, chunks)


# ---------------------------------------------------------------------------------------------
# (a) open-loop prediction


def openloop_ends(ep: Episode, k: int, e: int) -> np.ndarray:
    """``[STARTS, HORIZON + 1]`` window ends of the open-loop starts of episode ``e`` of test
    set ``k``, drawn with ``rng([k, e])``; column 0 is the start."""
    stride = model_stride(ep)
    last = ep.length - 1 - HORIZON * stride
    rng = np.random.default_rng([k, e])
    starts = rng.choice(np.arange(first_end(ep), last + 1), STARTS, replace=False)
    return sliding_ends(ep.length, first_end(ep), starts, stride, HORIZON)


def copy_arm() -> tuple[np.ndarray, np.ndarray]:
    """``[n, HORIZON]`` arm RMS on the open-loop starts of two model-free baselines, read from
    the episodes' proprio: copy (the start's arm held for every horizon) and the constant mean
    arm of these targets (what a decoder that reads nothing from the state would score)."""
    targets, rows = [], []
    for k, test in enumerate(TEST_SETS):
        for e, p in enumerate(variety.files(test, "heldout")):
            ep = Episode.load(p, privileged=False)
            ends = openloop_ends(ep, k, e)
            q = arm_q(ep, ends.ravel()).reshape(*ends.shape, -1)
            rows.append(rms(np.broadcast_to(q[:, :1], q[:, 1:].shape), q[:, 1:]))
            targets.append(q[:, 1:])
    q = np.concatenate(targets)
    return np.concatenate(rows), rms(q.reshape(-1, q.shape[-1]).mean(0), q)


def openloop(shard: str) -> None:
    """Per encoder and predictor: latent squared error, copy baseline and decoded arm RMS at
    every horizon, on ``STARTS`` starts of every held-out episode of both sets."""
    out = ROOT / "openloop"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, a) for p in ("own", "ridge") for a in (OWN if p == "own" else RIDGE)]
    jobs = [(p, a, s) for p, a in jobs for s in SEEDS]
    i, n = (int(x) for x in shard.split("/"))
    for predictor, arm, seed in jobs[i::n]:
        path = out / f"{predictor}-{arm}-s{seed}.npz"
        if path.exists():
            continue
        lay, dec = layer(arm, seed), heads(arm, seed)[1]
        rows: dict[str, list[np.ndarray]] = defaultdict(list)
        for k, test in enumerate(TEST_SETS):
            for e, p in enumerate(variety.files(test, "heldout")):
                ep = Episode.load(p, privileged=False)
                ends = openloop_ends(ep, k, e)
                starts = ends[:, 0]
                states = states_at(lay, ep, ends.ravel()).reshape(len(starts), HORIZON + 1, -1)
                flat = chunks_at(lay, ep, ends[:, :-1].ravel())
                chunks = flat.reshape(len(starts), HORIZON, *flat.shape[1:])
                pred = rollout(predictor, arm, seed, states[:, 0], chunks)
                true = states[:, 1:]
                q = arm_q(ep, ends[:, 1:].ravel()).reshape(len(starts), HORIZON, -1)
                rows["latent"].append(((pred - true) ** 2).mean(-1))
                rows["copy"].append(((states[:, :1] - true) ** 2).mean(-1))
                rows["decoded"].append(
                    rms(dec(pred.reshape(-1, pred.shape[-1])).reshape(q.shape), q)
                )
                rows["decoded_true"].append(
                    rms(dec(true.reshape(-1, true.shape[-1])).reshape(q.shape), q)
                )
                rows["true"].append(true.reshape(-1, true.shape[-1]))
                rows["groups"].append(np.full(len(starts), k * 1000 + e))
                rows["set"].append(np.full(len(starts), k))
        np.savez(path, **{k: np.concatenate(v) for k, v in rows.items()})
        logging.info("open loop: %s %s-s%d", predictor, arm, seed)


# ---------------------------------------------------------------------------------------------
# (b) goal reaching


def goal_list() -> list[tuple[str, int, Path, int]]:
    """Every goal: ``(test set, episode index, path, t0)``, ``GOALS_PER_EPISODE`` per held-out
    episode, drawn with ``rng([set, episode, 1])``."""
    goals = []
    for k, test in enumerate(TEST_SETS):
        for e, p in enumerate(variety.files(test, "heldout")):
            with np.load(p) as raw:
                length = int(raw["action"].shape[0])
            rng = np.random.default_rng([k, e, 1])
            t0s = rng.choice(
                np.arange(4, length - 1 - HORIZON * 4), GOALS_PER_EPISODE, replace=False
            )
            goals += [(test, e, p, int(t)) for t in sorted(t0s)]
    return goals


def configs() -> list[tuple[str, str, int]]:
    """``(predictor, arm, seed)`` of every planner, then the bars (seed 0 only)."""
    out = [("own", a, s) for a in OWN for s in SEEDS]
    out += [("ridge", a, s) for a in RIDGE for s in SEEDS]
    return out + [("bar", b, 0) for b in BARS]


@dataclass
class Planner:
    """One goal's latent planning problem: an encoder and predictor, the start and goal states,
    the held joint targets and the arm columns that are planned."""

    predictor: str
    arm: str
    seed: int
    header: Header
    goal: np.ndarray
    hold: np.ndarray
    cols: np.ndarray
    stride: int

    def cost(self, state: np.ndarray, cands: np.ndarray) -> np.ndarray:
        """Squared latent distance to the goal after the candidate arm targets ``[N, R, 8]``
        (one target per chunk, held for its ``stride`` steps; the other joints held)."""
        n, r = cands.shape[:2]
        full = np.broadcast_to(self.hold, (n, r, self.stride, len(self.hold))).copy()
        full[..., self.cols] = cands[:, :, None, :]
        chunks = union_actions(layer(self.arm, self.seed), self.header, full)
        start = np.repeat(state, n, 0)
        final = rollout(self.predictor, self.arm, self.seed, start, chunks)[:, -1]
        return np.asarray(((final - self.goal) ** 2).sum(-1))


def run_goal(
    scene: RestoredScene,
    ep: Episode,
    t0: int,
    config: tuple[str, str, int],
    stats: dict[str, dict[str, list[float]]],
) -> float:
    """Restore the scene at ``t0``, act for ``HORIZON`` model steps (MPC: plan, execute one
    chunk, re-encode), return the final arm RMS (radians) to the goal's arm-joint positions."""
    predictor, arm, seed = config
    stride, cols = model_stride(ep), arm_action_cols(ep)
    eff = stats[str(ep.meta.get("effector", "hand"))]
    std, low, high = (np.asarray(eff[k]) for k in ("std", "low", "high"))
    rng = np.random.default_rng([t0, len(arm), seed])
    scene.restore(ep, t0)
    hold = scene.hold.copy()
    end = t0 + HORIZON * stride
    goal_q = arm_q(ep, np.array([end]))[0]
    planner, state = None, np.zeros(0)
    if predictor != "bar":
        lay = layer(arm, seed)
        state = states_at(lay, ep, np.array([t0]))
        goal = states_at(lay, ep, np.array([end]))[0]
        header = load_episode(ep, lay.space, ()).header
        planner = Planner(predictor, arm, seed, header, goal, hold, cols, stride)
    for i in range(HORIZON):
        actions = np.tile(hold, (stride, 1))
        if arm == "replay":
            actions = np.asarray(ep.action)[t0 + i * stride : t0 + (i + 1) * stride]
        elif arm == "random":
            actions[:, cols] = np.clip(hold[cols] + std * rng.standard_normal(len(cols)), low, high)
        elif planner is not None:
            remaining = HORIZON - i
            mean = np.tile(hold[cols], (remaining, 1))
            plan = cem(
                partial(planner.cost, state), mean, np.tile(std, (remaining, 1)), low, high, rng
            )
            actions[:, cols] = plan[0]
        for a in actions:
            scene.step(a)
        observed = scene.observed()
        if planner is not None:
            state = states_at(layer(arm, seed), observed, np.array([observed.length - 1]))
    return float(rms(scene.arm_q(), goal_q))


Goal = tuple[str, int, Path, int]


def block_of(goal: Goal) -> tuple[str, str]:
    """The (test set, generator block) of a goal: the scene it needs."""
    return goal[0], goal[2].name.split("-")[1]


def shard_goals(goals_: list[Goal], shard: str) -> list[tuple[int, Goal]]:
    """This shard's goals, with their ids (the index in ``goals_``), grouped by block: whole
    blocks are dealt to shards in turn, so a process builds one scene per block it owns, and
    its goals are ordered by block, episode and ``t0``."""
    i, n = (int(x) for x in shard.split("/"))
    blocks = sorted({block_of(g) for g in goals_})
    mine = set(blocks[i::n])
    picked = [(gid, g) for gid, g in enumerate(goals_) if block_of(g) in mine]
    return sorted(picked, key=lambda item: (block_of(item[1]), item[1][2].name, item[1][3]))


def done_keys(folder: Path) -> set[tuple[str, int]]:
    """``(config, goal)`` of every result in any shard file of ``folder`` (so a change of
    sharding resumes correctly); a line cut short by a crash is ignored."""
    keys = set()
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((row["config"], row["goal"]))
    return keys


def peak_rss_mb() -> float:
    """This process's peak resident memory (MB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KB on Linux


def goals(shard: str, limit: int | None = None, only: tuple[str, ...] = ()) -> Path:
    """Run this shard's goals for every config (or ``only`` these bars; ``limit``: only the
    first goals by id); results are appended to ``runs/plan/goals/<shard>.jsonl``, and every
    result already in any shard file is skipped. One scene is alive at a time: it is closed
    when the next goal needs another block."""
    out = ROOT / "goals"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{shard.replace('/', '-')}{'-' + '-'.join(only) if only else ''}.jsonl"
    done = done_keys(out)
    stats = json.loads((ROOT / "fit" / "action_stats.json").read_text())
    all_goals = goal_list()
    mine = shard_goals(all_goals[:limit] if limit else all_goals, shard)
    chosen = [c for c in configs() if not only or (c[0] == "bar" and c[1] in only)]
    scene: RestoredScene | None = None
    current: tuple[str, str] | None = None
    with path.open("a") as log:
        for gid, goal in mine:
            todo = [c for c in chosen if ("/".join(map(str, c)), gid) not in done]
            if not todo:
                continue
            test, _, p, t0 = goal
            if block_of(goal) != current:
                if scene is not None:
                    scene.close()
                current = block_of(goal)
                scene = RestoredScene(block_settings(CONFIGS[test], current[1]))
            assert scene is not None
            ep = Episode.load(p)
            for config in todo:
                began = time.time()
                final = run_goal(scene, ep, t0, config, stats)
                row = {"config": "/".join(map(str, config)), "goal": gid, "set": test, "rms": final}
                log.write(json.dumps(row | {"seconds": round(time.time() - began, 2)}) + "\n")
                log.flush()
            logging.info(
                "goal %d done (%s t0 %d), peak RSS %.0f MB", gid, p.name, t0, peak_rss_mb()
            )
    if scene is not None:
        scene.close()
    return path


def fidelity() -> bool:
    """Stage 1: replay the true actions on the first ``FIDELITY_GOALS`` goals; the gate passes
    if at least ``FIDELITY_BAR`` of them end within ``SUCCESS_RAD``."""
    path = goals("0/1", FIDELITY_GOALS, ("replay",))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rate = float(np.mean([r["rms"] < SUCCESS_RAD for r in rows]))
    result = {
        "goals": len(rows),
        "replay_success": rate,
        "median_rms": float(np.median([r["rms"] for r in rows])),
        "seconds_per_goal": float(np.mean([r["seconds"] for r in rows])),
        "peak_rss_mb": round(peak_rss_mb()),
        "pass": rate >= FIDELITY_BAR,
    }
    (ROOT / "fidelity.json").write_text(json.dumps(result, indent=1))
    logging.info("fidelity: %s", result)
    return bool(result["pass"])


# ---------------------------------------------------------------------------------------------
# proxy: goal-conditioned action ranking


def rank(shard: str) -> None:
    """Rank each goal's true action sequence against ``CANDIDATES - 1`` others (true sequences
    of other goals with the same effector) by predicted final latent distance to the goal."""
    out = ROOT / "rank"
    out.mkdir(parents=True, exist_ok=True)
    goals_ = goal_list()
    episodes = {p: Episode.load(p, privileged=False) for _, _, p, _ in goals_}
    seqs = {
        gid: np.asarray(episodes[p].action)[t0 : t0 + HORIZON * 4].reshape(HORIZON, 4, -1)
        for gid, (_, _, p, t0) in enumerate(goals_)
    }
    jobs = [c for c in configs() if c[0] != "bar"]
    i, n = (int(x) for x in shard.split("/"))
    for predictor, arm, seed in jobs[i::n]:
        path = out / f"{predictor}-{arm}-s{seed}.npz"
        if path.exists():
            continue
        lay = layer(arm, seed)
        ranks = []
        for gid, (_, _, p, t0) in enumerate(goals_):
            ep = episodes[p]
            same = [g for g in seqs if seqs[g].shape == seqs[gid].shape and g != gid]
            others = np.random.default_rng(gid).choice(same, CANDIDATES - 1, replace=False)
            cands = np.stack([seqs[gid], *(seqs[g] for g in others)])
            header = load_episode(ep, lay.space, ()).header
            chunks = union_actions(lay, header, cands)
            start = states_at(lay, ep, np.array([t0]))
            goal = states_at(lay, ep, np.array([t0 + HORIZON * 4]))[0]
            final = rollout(predictor, arm, seed, np.repeat(start, len(cands), 0), chunks)[:, -1]
            cost = ((final - goal) ** 2).sum(-1)
            ranks.append(int((cost < cost[0]).sum()) + 1)
        np.savez(path, ranks=np.array(ranks))
        logging.info("ranked %s %s-s%d", predictor, arm, seed)


# ---------------------------------------------------------------------------------------------
# report


def reading(c: Contrast | None, lower_is_better: bool = False) -> str:
    """``better`` / ``worse`` / ``no difference`` for fix1 against the reference."""
    if c is None:
        return "incomplete"
    if not c.resolved:
        return "no difference"
    good = c.point < 0 if lower_is_better else c.point > 0
    return "fix1 better" if good else "fix1 worse"


def load_openloop(predictor: str, arm: str) -> list[dict[str, np.ndarray]]:
    """The open-loop results of an encoder's seeds (those present)."""
    out = []
    for s in SEEDS:
        path = ROOT / "openloop" / f"{predictor}-{arm}-s{s}.npz"
        if path.exists():
            raw = np.load(path)
            out.append({k: raw[k] for k in raw.files})
    return out


def report_openloop() -> None:
    """(a): normalised latent error, copy baseline and decoded arm RMS per horizon; primary
    contrast fix1 - A0 decoded RMS at horizon 8 (own predictors, pooled sets)."""
    variety.say("\n## (a) Open-loop prediction (mean +- sd over seeds)\n")
    cols = [h - 1 for h in HORIZONS]
    variety.say(
        "| predictor | arm | "
        + " | ".join(f"latent/var h{h}" for h in HORIZONS)
        + " | copy/var h8 | "
        + " | ".join(f"arm RMS h{h} (rad)" for h in HORIZONS)
        + " |"
    )
    variety.say("|---|---|" + "---|" * (2 * len(HORIZONS) + 1))
    curves = {}
    for predictor, arms in (("own", OWN), ("ridge", RIDGE)):
        for arm in arms:
            runs = load_openloop(predictor, arm)
            if not runs:
                continue
            var = [r["true"].var(0).mean() for r in runs]
            lat = np.array([r["latent"].mean(0) / v for r, v in zip(runs, var, strict=True)])
            cop = np.array([r["copy"].mean(0) / v for r, v in zip(runs, var, strict=True)])
            dec = np.array([r["decoded"].mean(0) for r in runs])
            curves[(predictor, arm)] = dec
            cells = [f"{lat[:, c].mean():.3f} +- {lat[:, c].std():.3f}" for c in cols]
            cells.append(f"{cop[:, -1].mean():.3f}")
            cells += [f"{dec[:, c].mean():.3f} +- {dec[:, c].std():.3f}" for c in cols]
            variety.say(f"| {predictor} | {arm} | " + " | ".join(cells) + " |")
    a, b = load_openloop("own", "fix1"), load_openloop("own", "A0")
    if len(a) == len(b) == len(SEEDS):
        groups = a[0]["groups"]
        c = contrast_values(
            np.stack([r["decoded"][:, -1] for r in a]),
            np.stack([r["decoded"][:, -1] for r in b]),
            groups,
        )
        verdict = reading(c, lower_is_better=True)
        variety.say(
            f"\n**Primary (a):** fix1 - A0 decoded arm RMS, horizon 8: {describe(c)}: {verdict}"
        )
    decoder_floor()
    fig, ax = plt.subplots(figsize=(6, 4))
    for (predictor, arm), dec in curves.items():
        ax.errorbar(
            HORIZONS,
            dec[:, cols].mean(0),
            dec[:, cols].std(0),
            marker="o",
            capsize=2,
            ls="-" if predictor == "own" else "--",
            label=f"{arm} ({predictor})",
        )
    ax.set(xscale="log", xlabel="horizon (model steps of 0.2 s)", ylabel="decoded arm RMS (rad)")
    ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
    ax.minorticks_off()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures" / "plan_openloop.png", dpi=130)
    plt.close(fig)


def decoder_floor() -> None:
    """The decoded arm RMS of the TRUE encoded state at each horizon (the decoder's floor: how
    well a state reads the arm at all), and the excess of the prediction over it, with the paired
    contrast fix1 - A0 of the excess (own predictors). The decoded RMS of the primary mixes the
    two; the excess is the part a predictor adds."""
    cols = [h - 1 for h in HORIZONS]
    curves = {}
    variety.say(
        "\nDecoder floor (decoded RMS of the true state) and excess (prediction - floor), rad\n"
    )
    head = (
        " | ".join(f"floor h{h}" for h in HORIZONS)
        + " | "
        + " | ".join(f"excess h{h}" for h in HORIZONS)
    )
    variety.say(f"| predictor | arm | {head} |")
    variety.say("|---|---|" + "---|" * (2 * len(HORIZONS)))
    for predictor, arms in (("own", OWN), ("ridge", RIDGE)):
        for arm in arms:
            runs = load_openloop(predictor, arm)
            if not runs:
                continue
            floor = np.array([r["decoded_true"].mean(0) for r in runs])
            excess = np.array([(r["decoded"] - r["decoded_true"]).mean(0) for r in runs])
            curves[(predictor, arm)] = (floor, floor + excess)
            cells = [f"{floor[:, c].mean():.3f}" for c in cols]
            cells += [f"{excess[:, c].mean():+.3f} +- {excess[:, c].std():.3f}" for c in cols]
            variety.say(f"| {predictor} | {arm} | " + " | ".join(cells) + " |")
    copy, mean = (b.mean(0) for b in copy_arm())
    for name, base in (("copy (true arm at the start)", copy), ("constant mean arm", mean)):
        cells = [f"{base[c]:.3f}" for c in cols] + ["" for _ in cols]
        variety.say(f"| {name} | - | " + " | ".join(cells) + " |")
    floor_figure(curves, copy[cols], mean[cols])
    a, b = load_openloop("own", "fix1"), load_openloop("own", "A0")
    if len(a) == len(b) == len(SEEDS):
        groups = a[0]["groups"]
        variety.say(
            "\n| fix1 - A0, own predictors (paired by seed and episode) | contrast |\n|---|---|"
        )
        for h in HORIZONS:
            c = contrast_values(
                np.stack([r["decoded"][:, h - 1] - r["decoded_true"][:, h - 1] for r in a]),
                np.stack([r["decoded"][:, h - 1] - r["decoded_true"][:, h - 1] for r in b]),
                groups,
            )
            variety.say(f"| excess, horizon {h} | {describe(c)} |")
        c = contrast_values(
            np.stack([r["decoded_true"][:, -1] for r in a]),
            np.stack([r["decoded_true"][:, -1] for r in b]),
            groups,
        )
        variety.say(f"| floor (true state), horizon 8 | {describe(c)} |")


def floor_figure(
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    copy: np.ndarray,
    mean: np.ndarray,
) -> None:
    """``figures/plan_floor.png``: per predictor, each arm's decoded RMS of the prediction (solid)
    and of the true state (dashed, the floor), with the model-free copy and constant baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    cols = [h - 1 for h in HORIZONS]
    for ax, predictor in zip(axes, ("own", "ridge"), strict=True):
        for (p, arm), (floor, pred) in curves.items():
            if p != predictor:
                continue
            color = f"C{list(ENCODERS).index(arm)}"  # one colour per encoder in both panels
            ax.errorbar(
                HORIZONS,
                pred[:, cols].mean(0),
                pred[:, cols].std(0),
                color=color,
                marker="o",
                capsize=2,
                label=f"{arm}: prediction",
            )
            ax.plot(
                HORIZONS,
                floor[:, cols].mean(0),
                color=color,
                ls="--",
                marker="x",
                label=f"{arm}: true state (floor)",
            )
        ax.plot(HORIZONS, copy, color="0.3", ls=":", label="copy the start's arm")
        ax.plot(HORIZONS, mean, color="0.6", ls="-.", label="constant mean arm")
        ax.set(
            xscale="log", xlabel="horizon (model steps of 0.2 s)", title=f"{predictor} predictor"
        )
        ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
        ax.minorticks_off()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("decoded arm RMS (rad)")
    fig.tight_layout()
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures" / "plan_floor.png", dpi=130)
    plt.close(fig)


def goal_table() -> tuple[dict[str, dict[int, float]], np.ndarray]:
    """``{config: {goal: rms}}`` from every shard file, and each goal's episode group."""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for path in sorted((ROOT / "goals").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[r["config"]][r["goal"]] = r["rms"]
    goals_ = goal_list()
    groups = np.array([TEST_SETS.index(t) * 1000 + e for t, e, _, _ in goals_])
    return out, groups


def success(
    table: dict[str, dict[int, float]], config: str, n: int, thr: float = SUCCESS_RAD
) -> np.ndarray | None:
    """``[n]`` success indicators of a planner config (``None`` until every goal is done)."""
    got = table.get(config, {})
    return np.array([got[g] < thr for g in range(n)], dtype=float) if len(got) >= n else None


def success_family(
    table: dict[str, dict[int, float]], predictor: str, arm: str, n: int, thr: float = SUCCESS_RAD
) -> np.ndarray | None:
    """``[seeds, n]`` success indicators (bars: seed 0 only), or ``None`` if incomplete."""
    seeds = (0,) if predictor == "bar" else SEEDS
    rows = [success(table, f"{predictor}/{arm}/{s}", n, thr) for s in seeds]
    return None if any(r is None for r in rows) else np.stack(rows)  # type: ignore[arg-type]


PLANNERS = [("own", a) for a in OWN] + [("ridge", a) for a in RIDGE] + [("bar", b) for b in BARS]
CONTRASTS = [
    ("Primary (b): fix1 - A0, own predictors", ("own", "fix1"), ("own", "A0")),
    ("fix1 - recon05, own predictors", ("own", "fix1"), ("own", "recon05")),
    ("fix1 - MAE, ridge heads", ("ridge", "fix1"), ("ridge", "MAE")),
    ("A0 own - A0 ridge", ("own", "A0"), ("ridge", "A0")),
    ("fix1 own - fix1 ridge", ("own", "fix1"), ("ridge", "fix1")),
]


def report_goals() -> None:
    """(b): success per planner, the contrasts and the curve over thresholds."""
    fid_path = ROOT / "fidelity.json"
    fid = json.loads(fid_path.read_text()) if fid_path.exists() else {}
    variety.say(f"\n## (b) Goal reaching (closed loop)\n\nFidelity gate: {fid}\n")
    table, groups = goal_table()
    n = len(groups)
    if any(success_family(table, p, a, n) is None for p, a in PLANNERS):
        diagnosis(table)
        return
    variety.say(f"| planner | success (< {SUCCESS_RAD} rad), mean +- sd | median final RMS |")
    variety.say("|---|---|---|")
    for predictor, arm in PLANNERS:
        fam = success_family(table, predictor, arm, n)
        if fam is None:
            variety.say(f"| {predictor} {arm} | incomplete | - |")
            continue
        seeds = (0,) if predictor == "bar" else SEEDS
        med = np.median([table[f"{predictor}/{arm}/{s}"][g] for s in seeds for g in range(n)])
        rates = fam.mean(1)
        variety.say(f"| {predictor} {arm} | {rates.mean():.3f} +- {rates.std():.3f} | {med:.3f} |")
    variety.say("\n| contrast (success, paired by seed and goal) | contrast | reading |")
    variety.say("|---|---|---|")
    for label, x, y in CONTRASTS:
        a, b = success_family(table, *x, n), success_family(table, *y, n)
        c = contrast_values(a, b, groups) if a is not None and b is not None else None
        variety.say(f"| {label} | {describe(c) if c else '-'} | {reading(c)} |")
    success_figure(table, n)


DIAGNOSIS_THRESHOLDS = (0.10, 0.30, 0.50)


def diagnosis(table: dict[str, dict[int, float]]) -> None:
    """Goal reaching with planners incomplete: no primary and no contrast. Instead a post hoc
    description on the goals that every planner and bar finished: success at a few thresholds
    and the median final arm RMS (planners: mean +- sd over seeds)."""
    keys = [f"{p}/{a}/{s}" for p, a in PLANNERS for s in ((0,) if p == "bar" else SEEDS)]
    sets = [set(table.get(k, {})) for k in keys]
    common = sorted(set.intersection(*sets)) if all(sets) else []
    variety.say(
        "**Incomplete: the pre-registered primary and contrasts are not computed.** Post hoc "
        f"planner-failure diagnosis on the {len(common)} goals every planner and bar finished "
        f"(of {max(len(x) for x in sets) if sets else 0} done by some):\n"
    )
    head = " | ".join(f"success < {t:.2f}" for t in DIAGNOSIS_THRESHOLDS)
    variety.say(f"| planner | {head} | median final RMS (rad) |")
    variety.say("|---|" + "---|" * (len(DIAGNOSIS_THRESHOLDS) + 1))
    if not common:
        return
    for predictor, arm in PLANNERS:
        seeds = (0,) if predictor == "bar" else SEEDS
        rms_ = np.array([[table[f"{predictor}/{arm}/{s}"][g] for g in common] for s in seeds])
        cells = []
        for t in DIAGNOSIS_THRESHOLDS:
            rates = (rms_ < t).mean(1)
            cells.append(
                f"{rates.mean():.3f}" + (f" +- {rates.std():.3f}" if len(seeds) > 1 else "")
            )
        meds = np.median(rms_, 1)
        med = f"{meds.mean():.2f}" + (
            f" ({meds.min():.2f}-{meds.max():.2f})" if len(seeds) > 1 else ""
        )
        variety.say(f"| {predictor} {arm} | " + " | ".join(cells) + f" | {med} |")


def success_figure(table: dict[str, dict[int, float]], n: int) -> None:
    """Success rate against the threshold, per planner."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for predictor, arm in PLANNERS:
        fams = [success_family(table, predictor, arm, n, float(t)) for t in CURVE]
        if any(f is None for f in fams):
            continue
        ys = [float(f.mean()) for f in fams]  # type: ignore[union-attr]
        style = ":" if predictor == "bar" else "-"
        ax.plot(CURVE, ys, marker="o", ms=3, ls=style, label=f"{predictor} {arm}")
    ax.axvline(SUCCESS_RAD, color="#888888", lw=1)
    ax.set(xlabel="success threshold (arm RMS, rad)", ylabel="success rate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "figures" / "plan_success.png", dpi=130)
    plt.close(fig)


def report_rank() -> None:
    """The proxy: top-1 rate and mean rank of the true action sequence per planner."""
    variety.say(
        f"\n## (b) proxy: goal-conditioned action ranking (chance top-1 = {1 / CANDIDATES:.3f})\n"
    )
    variety.say("| planner | top-1 | mean rank |\n|---|---|---|")
    for predictor, arm, _ in [c for c in configs() if c[0] != "bar" and c[2] == 0]:
        runs = [ROOT / "rank" / f"{predictor}-{arm}-s{s}.npz" for s in SEEDS]
        if not all(p.exists() for p in runs):
            continue
        r = [np.load(p)["ranks"] for p in runs]
        top1, mean_rank = np.mean([(x == 1).mean() for x in r]), np.mean([x.mean() for x in r])
        variety.say(f"| {predictor} {arm} | {top1:.3f} | {mean_rank:.1f} |")


# ---------------------------------------------------------------------------------------------
# post hoc: is the weak arm readout the representation or the decoder's fit?

DECODER_FOLDS = 5
DECODER_FEATURES = ("z", "s")


def decoder_samples(arm: str, seed: int, paths: list[Path]) -> dict[str, np.ndarray]:
    """Every model step of the episodes at ``paths``: the arm's features (summary token ``z``
    and state ``s``, or for ``arm == "raw"`` the proprio of the window's ``substeps`` steps on
    the channels every episode has), the arm-joint targets, episode groups and effectors."""
    feats: dict[str, list[np.ndarray]] = defaultdict(list)
    raw: list[dict[str, np.ndarray]] = []
    q, groups, effector = [], [], []
    for i, p in enumerate(paths):
        ep = Episode.load(p, privileged=False)
        ends = np.arange(first_end(ep), ep.length, model_stride(ep))
        if arm == "raw":
            back = spacing(ep.dt) * np.arange(4)  # the window's substeps, newest first
            window = np.asarray(ep.proprio)[ends[:, None] - back[None]]  # [n, 4, channels]
            raw.append(dict(zip(ep.proprio_names, np.moveaxis(window, 2, 0), strict=True)))
        else:
            s, z = encode_at(layer(arm, seed), ep, ends)
            feats["s"].append(s)
            feats["z"].append(z)
        q.append(arm_q(ep, ends))
        groups.append(np.full(len(ends), i))
        effector.append(np.full(len(ends), str(ep.meta.get("effector", "hand"))))
    if arm == "raw":
        names = sorted(set.intersection(*(set(r) for r in raw)))
        feats["raw"] = [np.concatenate([r[n] for n in names], 1) for r in raw]
    return {
        **{k: np.concatenate(v) for k, v in feats.items()},
        "q": np.concatenate(q),
        "groups": np.concatenate(groups),
        "effector": np.concatenate(effector),
    }


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    """1 - SSE / SST over all joints together (SST around ``y``'s own mean)."""
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def fit_decoder(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, eff: np.ndarray, per: bool
) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], list[float]]:
    """A predict function: one ridge map (pooled), or one per effector (``per``; an effector
    not seen in fitting falls back to the pooled map). Also returns the strengths over d."""
    pooled = Ridge.fit(x, y, groups)
    maps = {e: Ridge.fit(x[eff == e], y[eff == e], groups[eff == e]) for e in np.unique(eff)}
    alphas = [m.alpha / x.shape[1] for m in (maps.values() if per else [pooled])]

    def predict(xs: np.ndarray, es: np.ndarray) -> np.ndarray:
        out = pooled(xs)
        if per:
            for e, m in maps.items():
                out[es == e] = m(xs[es == e])
        return out

    return predict, alphas


def decoder_scores(
    fit_: dict[str, np.ndarray], held: dict[str, dict[str, np.ndarray]], feature: str
) -> dict[str, dict[str, float]]:
    """Per decoder (pooled, per effector): train R2, episode-grouped CV R2 on the fit episodes,
    held-out R2 per test set and the strength (ALPHA_GRID x d)."""
    x, y, g, e = fit_[feature], fit_["q"], fit_["groups"], fit_["effector"]
    out = {}
    for name, per in (("pooled", False), ("per effector", True)):
        predict, alphas = fit_decoder(x, y, g, e, per)
        cv = np.zeros_like(y)
        for tr, te in episode_folds(g, DECODER_FOLDS):
            fold, _ = fit_decoder(x[tr], y[tr], g[tr], e[tr], per)
            cv[te] = fold(x[te], e[te])
        row = {"train": r2(predict(x, e), y), "cv": r2(cv, y), "alpha": float(np.mean(alphas))}
        for test, h in held.items():
            row[test] = r2(predict(h[feature], h["effector"]), h["q"])
        out[name] = row
    return out


def decoder(shard: str) -> None:
    """Post hoc decoder check: R2 of ridge arm decoders on z, s and the raw proprio window, fitted
    on the ``fit`` episodes, scored on them (train, CV) and on both held-out sets."""
    out = ROOT / "decoder"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("raw", 0), *((a, s) for a in ENCODERS for s in SEEDS)]
    i, n = (int(x) for x in shard.split("/"))
    for arm, seed in jobs[i::n]:
        path = out / f"{arm}-s{seed}.json"
        if path.exists():
            continue
        fit_ = decoder_samples(arm, seed, fit_paths())
        held = {t: decoder_samples(arm, seed, variety.files(t, "heldout")) for t in TEST_SETS}
        features = ("raw",) if arm == "raw" else DECODER_FEATURES
        result = {f: decoder_scores(fit_, held, f) for f in features}
        path.write_text(json.dumps(result, indent=1))
        logging.info("decoder check: %s-s%d, peak RSS %.0f MB", arm, seed, peak_rss_mb())


def decoder_report() -> None:
    """The decoder check's table: R2 per encoder, feature and decoder (mean +- sd over seeds)."""
    variety.say("\n## Post hoc: arm decoder R2 (fit on 100 babble_v2 pretraining episodes)\n")
    cols = ("train", "cv", *TEST_SETS)
    variety.say(
        "| encoder | feature | decoder | train | CV (5 episode folds) | held-out babble "
        "| held-out variety | strength / d |"
    )
    variety.say("|---|---|---|---|---|---|---|---|")
    for arm in ("raw", *ENCODERS):
        seeds = (0,) if arm == "raw" else SEEDS
        paths = [ROOT / "decoder" / f"{arm}-s{s}.json" for s in seeds]
        runs = [json.loads(p.read_text()) for p in paths if p.exists()]
        if not runs:
            continue
        for feature in runs[0]:
            for name in runs[0][feature]:
                v = {c: np.array([r[feature][name][c] for r in runs]) for c in (*cols, "alpha")}
                cells = [
                    f"{v[c].mean():.3f}" + (f" +- {v[c].std():.3f}" if len(runs) > 1 else "")
                    for c in cols
                ]
                cells.append(f"{v['alpha'].mean():.2g}")
                label = "raw proprio window" if arm == "raw" else arm
                variety.say(f"| {label} | {feature} | {name} | " + " | ".join(cells) + " |")


def report() -> None:
    """Every table and figure."""
    variety.say("B-plan-0 (pre-registered in docs/model.md).")
    report_openloop()
    if (ROOT / "goals").exists():
        report_goals()
    if (ROOT / "rank").exists():
        report_rank()


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=(
            *("fit", "fidelity", "openloop", "goals", "rank", "report"),
            *("decoder", "decoder-report"),
        ),
    )
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "fit":
        fit()
    elif args.command == "fidelity":
        raise SystemExit(0 if fidelity() else 3)  # exit 3: the gate failed (use the proxy)
    elif args.command == "openloop":
        openloop(args.shard)
    elif args.command == "goals":
        goals(args.shard)
    elif args.command == "rank":
        rank(args.shard)
    elif args.command == "decoder":
        decoder(args.shard)
    elif args.command == "decoder-report":
        decoder_report()
    else:
        report()


if __name__ == "__main__":
    main()
